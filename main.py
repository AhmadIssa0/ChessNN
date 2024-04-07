
import torch
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import torch.nn as nn
from lichess_dataset import FEN_CHAR_TO_INDEX, expand_fen_string, MAX_FEN_LENGTH, JSONLinesChessDataset, collate_fn, JSONLinesLichessDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import glob
import os
import random
from torch.cuda.amp import autocast
from tqdm import tqdm
from bin_predictor import BinPredictor
from functools import partial


def remove_full_half_moves_from_fen(fen):
    parts = fen.split(' ')
    if len(parts) == 6:
        return ' '.join(parts[:-2])
    else:
        return fen


class ChessTranformer(nn.Module):

    def __init__(self, bin_predictor: BinPredictor, d_model=512, num_layers=8, nhead=8, dim_feedforward=2048):
        super(ChessTranformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(FEN_CHAR_TO_INDEX), embedding_dim=d_model)
        self.proj_to_win_prob_logit = nn.Linear(d_model, 1)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation="gelu")
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # Use a learned positional embedding
        self.pos_embedding = nn.Embedding(num_embeddings=MAX_FEN_LENGTH + 1, embedding_dim=d_model)

        self.proj_to_bin_predictor_logits = nn.Linear(d_model, bin_predictor.total_num_bins)

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bin_predictor = bin_predictor

    def compute_cross_entropy_loss(self, emb_indices: torch.LongTensor, bin_pred_classes: torch.LongTensor):
        logits = self.compute_bin_predictor_logits(emb_indices)
        return self.cross_entropy_loss(logits, bin_pred_classes)  # (B)

    def compute_bin_predictor_logits(self, emb_indices: torch.LongTensor):
        embedding = self.embedding(emb_indices)  # (B, MAX_FEN_LENGTH + 1, d_model)
        embedding = embedding + self.pos_embedding.weight.unsqueeze(0)
        output = self.transformer(embedding)
        # Only look at the first token's output (i.e. eval token)
        return self.proj_to_bin_predictor_logits(self.layer_norm(output[:, 0, :]))  # (B, total_num_bins)

    def compute_bin_probabilities(self, emb_indices: torch.LongTensor):
        return self.compute_bin_predictor_logits(emb_indices).softmax(dim=1)

    def compute_bin_probabilities_from_fens(self, fen_list, device):
        return self.compute_bin_probabilities(self.fen_list_to_emb_indices(fen_list, device))

    def compute_avg_bin_index(self, emb_indices: torch.LongTensor):
        class_probs = self.compute_bin_probabilities(emb_indices)
        class_indices = torch.arange(0, self.bin_predictor.total_num_bins, device=emb_indices.device).unsqueeze(0)
        return (class_probs * class_indices).sum(dim=1)

    def compute_bin_index_means_and_stds_from_fens(self, fen_list, device):
        probabilities = self.compute_bin_probabilities_from_fens(fen_list, device)
        indices = torch.arange(0, self.bin_predictor.total_num_bins, device=device).unsqueeze(0)

        mean_indices = torch.sum(indices * probabilities, dim=1)
        squared_diffs = (indices - mean_indices.unsqueeze(1)) ** 2
        variances = torch.sum(squared_diffs * probabilities, dim=1)
        std_deviations = torch.sqrt(variances)
        return mean_indices, std_deviations

    # def forward(self, emb_indices: torch.LongTensor):
    #     """Same as compute_avg_bin_index()"""
    #     class_probs = self.compute_bin_probabilities(emb_indices)
    #     class_indices = torch.arange(0, self.bin_predictor.total_num_bins, device=emb_indices.device).unsqueeze(0)
    #     return (class_probs * class_indices).sum(dim=1)

    def compute_avg_bin_index_from_fens(self, fen_list, device):
        return self.compute_avg_bin_index(self.fen_list_to_emb_indices(fen_list, device))

    def fen_list_to_emb_indices(self, fen_list, device) -> torch.LongTensor:
        emb_indices_batch = []
        for fen_str in fen_list:
            expanded_fen = remove_full_half_moves_from_fen(expand_fen_string(fen_str))
            # print(expanded_fen)
            indices_lst = [FEN_CHAR_TO_INDEX['eval']] + [FEN_CHAR_TO_INDEX[c] for c in expanded_fen]
            # Pad to 85 characters
            indices_lst += [FEN_CHAR_TO_INDEX[' '] for _ in range(MAX_FEN_LENGTH - len(expanded_fen))]
            emb_indices_batch.append(
                torch.tensor(indices_lst, dtype=torch.long)
            )
        return torch.stack(emb_indices_batch, dim=0).to(device=device)

    def compute_white_win_prob_from_fen(self, fen_list, device):
        emb_indices = self.fen_list_to_emb_indices(fen_list, device)
        return self.compute_white_win_prob(emb_indices)

    def compute_white_win_prob(self, emb_indices: torch.LongTensor):
        embedding = self.embedding(emb_indices)  # (B, MAX_FEN_LENGTH + 1, d_model)
        embedding = embedding + self.pos_embedding.weight.unsqueeze(0)
        output = self.transformer(embedding)
        return self.proj_to_win_prob_logit(self.layer_norm(output[:, 0, :])).squeeze(-1).sigmoid()


def evaluate_prob_of_win(transformer, dataloader, dataset_name, loss_fn, global_step, summary_writer):
    with torch.no_grad():
        total_test_samples = 0
        acc_loss = 0.0
        for test_batch in dataloader:
            test_batch = test_batch.to(device=device)
            pred_white_win_prob = transformer.compute_white_win_prob(test_batch.embedding_indices)
            effective_batch_size = test_batch.cp_valid.sum().item()
            # total_test_samples += effective_batch_size
            total_test_samples += len(test_batch.cp_valid)
            # acc_loss += ((pred_evals - test_batch.cp_evals / 100.0) ** 2)[test_batch.cp_valid].sum().item()
            acc_loss += loss_fn(pred_white_win_prob, test_batch.white_win_prob).sum().item()
        loss = acc_loss / total_test_samples
        print(
            f'{dataset_name} loss: {loss}, Total samples: {total_test_samples}')
        summary_writer.add_scalar(
            f'Loss/{dataset_name}',
            loss,
            global_step
        )
        print('Pred_win_prob:', pred_white_win_prob[:10])
        print('True_win_prob:', test_batch.white_win_prob[:10])
        print('Valid?', test_batch.cp_valid[:10])


def evaluate_cross_entropy(transformer, dataloader, dataset_name, global_step, summary_writer):
    with torch.no_grad():
        total_test_samples = 0
        acc_loss = 0.0
        for test_batch in dataloader:
            test_batch = test_batch.to(device=device)
            total_test_samples += len(test_batch.cp_valid)
            acc_loss += transformer.compute_cross_entropy_loss(
                emb_indices=test_batch.embedding_indices, bin_pred_classes=test_batch.bin_classes
            ).sum().item()
        loss = acc_loss / total_test_samples
        print(
            f'{dataset_name} loss: {loss}, Total samples: {total_test_samples}')
        summary_writer.add_scalar(
            f'CrossEntropyLoss/{dataset_name}',
            loss,
            global_step
        )
        # print('Pred_win_prob:', pred_white_win_prob[:10])
        # print('True_win_prob:', test_batch.white_win_prob[:10])
        # print('Valid?', test_batch.cp_valid[:10])


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # For all GPUs, if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # To ensure that CUDA's convolution operations are deterministic.
    torch.backends.cudnn.benchmark = False  # False for reproducibility, True may improve performance if inputs size does not vary.


def train_to_predict_prob_of_win():
    device = 'cuda'
    # filepath = r'/mnt/c/Users/Ahmad-personal/Downloads/lichess_db_eval.jsonl'
    # filepath = r'C:/Users/Ahmad-personal/Downloads/lichess_db_eval.jsonl'
    # dataset = JSONLinesChessDataset(filepath)

    # print(bin_pred.to_bin_index(10, 3))
    # print(bin_pred.bin_index_to_description(43))



    filepath = r"C:\Users\Ahmad-personal\PycharmProjects\chess_stackfish_evals\data\lichess_db_standard_rated_2017-05.jsonl"
    dataset = JSONLinesLichessDataset(filepath)
    print('Dataset size:', len(dataset))
    print(dataset.__getitem__(11000))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cp_evals = []
    mates = []
    for i in range(2000000):
        raw_io = dataset.__getitem__(i)
        cp_evals.append(raw_io.cp_eval)
        if not raw_io.contains_eval:
            mates.append(raw_io.mate)


    plt.figure(figsize=(10, 6))
    plt.hist(mates, bins=200, color='skyblue', alpha=0.7, rwidth=0.85)
    plt.title('Stockfish mate in x, lichess dataset')
    plt.xlabel('Number of moves to mate')
    plt.ylabel('Frequency')
    plt.savefig('mates.png')

    import numpy as np
    cp_evals_sorted = np.sort(cp_evals)
    percentiles = np.percentile(cp_evals_sorted, 0.5 * np.arange(1, 201))
    print(sorted(list(set(percentiles.tolist()))))
    exit(0)

    set_seed(42)

    # Calculate the sizes of each dataset
    dataset_size = len(dataset)
    test_size = 25000
    train_size = dataset_size - test_size

    # Create the train-test split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    subset_indices = torch.randperm(len(train_dataset))[:25000]
    subset_dataset = Subset(train_dataset, subset_indices)
    set_seed(49)
    eval_batch_size = 1024
    bin_pred = BinPredictor()
    collate_fn = partial(collate_fn, bin_predictor=bin_pred)

    train_eval_dataloader = DataLoader(
        subset_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=5,
        collate_fn=collate_fn, pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=512, shuffle=True, num_workers=5, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, prefetch_factor=2, persistent_workers=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=5, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True
    )

    d_model = 512
    transformer = ChessTranformer(d_model=d_model, num_layers=8, nhead=8, dim_feedforward=4*d_model).to(device=device)
    scaler = torch.cuda.amp.GradScaler()
    optim = Adam(transformer.parameters(), lr=0.0001)

    global_step = 369500
    if True:
        checkpoint_path = f'checkpoint_{global_step}.pth'
        checkpoint = torch.load(checkpoint_path)
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']

    for param_group in optim.param_groups:
        param_group['lr'] = 0.00005

    summary_writer = SummaryWriter('runs/dModel_512_nlayers_8_nhead_8')
    l2_loss_fn = torch.nn.MSELoss(reduction='none')

    # batch_iterator = skip_to_batch(train_dataloader, global_step)
    for batch in train_dataloader:
        global_step += 1
        optim.zero_grad()
        with autocast():
            batch = batch.to(device=device)

            pred_white_win_prob = transformer.compute_white_win_prob(batch.embedding_indices)
            effective_batch_size = batch.cp_valid.sum().item()
            # acc_loss += ((pred_evals - test_batch.cp_evals / 100.0) ** 2)[test_batch.cp_valid].sum().item()
            loss = l2_loss_fn(pred_white_win_prob, batch.white_win_prob).mean()

        # Scales the loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        max_norm = 0.3
        scaler.unscale_(optim)  # Unscales the gradients of optimizer's assigned params in-place
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm)

        # If gradients do not contain inf or NaN, updates the parameters
        scaler.step(optim)
        scaler.update()

        if global_step % 100 == 0:
            print('Global step:', global_step)
            summary_writer.add_scalar(
                f'Loss/GradientNorm',
                grad_norm,
                global_step
            )
            summary_writer.add_scalar(
                f'Loss/TrainLoss',
                loss,
                global_step
            )

        if global_step % 1000 == 0:
            transformer.eval()
            evaluate_prob_of_win(transformer, test_dataloader, 'Test-Set', l2_loss_fn, global_step, summary_writer)
            evaluate_prob_of_win(transformer, train_eval_dataloader, 'Train-Eval-Set', l2_loss_fn, global_step, summary_writer)
            transformer.train()

        if global_step % 500 == 0:
            # Define the pattern for the checkpoint files
            checkpoint_pattern = "checkpoint_*.pth"

            # List all files matching the checkpoint pattern
            existing_checkpoints = glob.glob(checkpoint_pattern)

            # Delete all existing checkpoint files
            for checkpoint_file in existing_checkpoints:
                os.remove(checkpoint_file)
                print(f'Deleted {checkpoint_file}')

            # Save the new checkpoint
            checkpoint_path = f'checkpoint_{global_step}.pth'

            torch.save(
                {
                    'transformer_state_dict': transformer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'global_step': global_step,
                }, checkpoint_path)
            print(f'Saved model at {checkpoint_path}.')


def convert_checkpoint_to_bin_predictor():
    d_model = 512
    bin_pred = BinPredictor()
    transformer = ChessTranformer(
        bin_predictor=bin_pred, d_model=d_model, num_layers=8, nhead=8, dim_feedforward=4 * d_model
    ).to(device=device)
    scaler = torch.cuda.amp.GradScaler()
    optim = Adam(transformer.parameters(), lr=0.0001)

    global_step = 424000
    if True:
        checkpoint_path = f'checkpoint_424000_l2.pth'
        checkpoint = torch.load(checkpoint_path)
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']

    transformer.proj_to_bin_predictor_logits = nn.Linear(d_model, bin_pred.total_num_bins)
    torch.save(
        {
            'transformer_state_dict': transformer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'global_step': global_step,
        }, 'checkpoint_424000.pth'
    )



if __name__ == '__main__':
    device = 'cuda'
    # filepath = r'/mnt/c/Users/Ahmad-personal/Downloads/lichess_db_eval.jsonl'
    # filepath = r'C:/Users/Ahmad-personal/Downloads/lichess_db_eval.jsonl'
    # dataset = JSONLinesChessDataset(filepath)

    # print(bin_pred.to_bin_index(10, 3))
    # print(bin_pred.bin_index_to_description(43))

    # filepath = r"C:\Users\Ahmad-personal\PycharmProjects\chess_stackfish_evals\data\lichess_db_standard_rated_2017-05.jsonl"
    filepath = r"C:\Users\Ahmad-personal\PycharmProjects\chess_stackfish_evals\data\lichess_db_standard_rated_2017-07.jsonl"
    dataset = JSONLinesLichessDataset(filepath)
    print('Dataset size:', len(dataset))
    print(dataset.__getitem__(11000))

    set_seed(42)

    # Calculate the sizes of each dataset
    dataset_size = len(dataset)
    test_size = 25000
    train_size = dataset_size - test_size

    # Create the train-test split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    subset_indices = torch.randperm(len(train_dataset))[:25000]
    subset_dataset = Subset(train_dataset, subset_indices)
    set_seed(1)
    eval_batch_size = 512
    bin_pred = BinPredictor()
    collate_fn = partial(collate_fn, bin_predictor=bin_pred)

    train_eval_dataloader = DataLoader(
        subset_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=5,
        collate_fn=collate_fn, pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=512, shuffle=True, num_workers=5, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, prefetch_factor=2, persistent_workers=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=5, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True
    )

    d_model = 512
    transformer = ChessTranformer(
        bin_predictor=bin_pred, d_model=d_model, num_layers=8, nhead=8, dim_feedforward=4*d_model
    ).to(device=device)
    scaler = torch.cuda.amp.GradScaler()
    optim = Adam(transformer.parameters(), lr=0.0001)

    global_step = 751000
    if True:
        checkpoint_path = f'checkpoint_{global_step}.pth'
        checkpoint = torch.load(checkpoint_path)
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']

    for param_group in optim.param_groups:
        param_group['lr'] = 0.0001

    summary_writer = SummaryWriter('runs/dModel_512_nlayers_8_nhead_8_binPredictor')

    # batch_iterator = skip_to_batch(train_dataloader, global_step)
    for batch in train_dataloader:
        global_step += 1
        optim.zero_grad()
        with autocast():
            batch = batch.to(device=device)
            loss = transformer.compute_cross_entropy_loss(
                emb_indices=batch.embedding_indices, bin_pred_classes=batch.bin_classes
            ).mean()

        # Scales the loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        max_norm = 0.3
        scaler.unscale_(optim)  # Unscales the gradients of optimizer's assigned params in-place
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm)

        # If gradients do not contain inf or NaN, updates the parameters
        scaler.step(optim)
        scaler.update()

        if global_step % 100 == 0:
            print('Global step:', global_step)
            summary_writer.add_scalar(
                f'CrossEntropyLoss/GradientNorm',
                grad_norm,
                global_step
            )
            summary_writer.add_scalar(
                f'CrossEntropyLoss/TrainLoss',
                loss,
                global_step
            )

        if global_step % 1000 == 0:
            transformer.eval()
            evaluate_cross_entropy(transformer, test_dataloader, 'Test-Set', global_step, summary_writer)
            evaluate_cross_entropy(transformer, train_eval_dataloader, 'Train-Eval-Set', global_step, summary_writer)
            transformer.train()

        if global_step % 500 == 0:
            # Define the pattern for the checkpoint files
            checkpoint_pattern = "checkpoint_*.pth"

            # List all files matching the checkpoint pattern
            existing_checkpoints = glob.glob(checkpoint_pattern)

            # Delete all existing checkpoint files
            for checkpoint_file in existing_checkpoints:
                os.remove(checkpoint_file)
                print(f'Deleted {checkpoint_file}')

            # Save the new checkpoint
            checkpoint_path = f'checkpoint_{global_step}.pth'

            torch.save(
                {
                    'transformer_state_dict': transformer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'global_step': global_step,
                }, checkpoint_path)
            print(f'Saved model at {checkpoint_path}.')



