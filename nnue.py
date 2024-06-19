
import torch
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import torch.nn as nn
from lichess_dataset import FEN_CHAR_TO_INDEX, expand_fen_string, MAX_FEN_LENGTH, JSONLinesChessDataset, nnue_collate_fn, JSONLinesLichessDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
import glob
import os
import random
from torch.cuda.amp import autocast
from tqdm import tqdm
from functools import partial
from main import set_seed
from lichess_dataset import fen_to_halfkp


class NNUE(nn.Module):

    def __init__(self, embedding_dim, num_hidden1, num_hidden2, num_embeddings=10*64*64 + 1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=10*64*64)

        self.layer1 = nn.Linear(2 * embedding_dim, num_hidden1)
        self.layer2 = nn.Linear(num_hidden1, num_hidden2)
        self.layer3 = nn.Linear(num_hidden2, 1)

    def forward(self, wb_emb_indices, side_to_move):
        """

        :param wb_emb_indices: (B, 2, 16)
        :return:
        """
        wb_features = self.embedding(wb_emb_indices).sum(dim=-2)
        w_features, b_features = wb_features[:, 0, :], wb_features[:, 1, :]
        side_to_move = side_to_move.unsqueeze(1)
        accumulator = (side_to_move * torch.concat([w_features, b_features], 1) +
                       (1 - side_to_move) * torch.concat([b_features, w_features], 1))
        hidden1 = self.layer1(torch.clamp(accumulator, 0.0, 1.0))
        hidden2 = self.layer2(torch.clamp(hidden1, 0.0, 1.0))
        side_to_move = side_to_move.squeeze(1)

        # layer3 outputs values so that positive means good for side to move.
        # but we want to deal with things so that positive means good for white
        return (side_to_move - 0.5) * 2.0 * self.layer3(torch.clamp(hidden2, 0.0, 1.0)).squeeze(-1)
        # return self.layer3(hidden2).squeeze(-1)

    def compute_white_win_prob(self, wb_emb_indices, side_to_move):
        output = self.forward(wb_emb_indices, side_to_move)
        scale = 0.00368208
        return (scale * output).sigmoid()

    @staticmethod
    def logit_to_white_win_prob(logits):
        scale = 0.00368208
        return (scale * logits).sigmoid()

    @staticmethod
    def fen_to_indices_stm(expanded_fen, device):
        white_indices, black_indices = fen_to_halfkp(expanded_fen)

        stm_chr = expanded_fen.split(' ')[1]
        assert(stm_chr in ['w', 'b'])
        stm = torch.tensor(1 if stm_chr == 'w' else 0).to(device=device)
        white_indices = white_indices + [10 * 64 * 64] * (16 - len(white_indices))
        black_indices = black_indices + [10 * 64 * 64] * (16 - len(black_indices))
        wb_emb_indices = torch.stack([
                torch.tensor(white_indices, dtype=torch.long),
                torch.tensor(black_indices, dtype=torch.long)
        ]).to(device=device)
        return wb_emb_indices, stm

    @staticmethod
    def fens_to_indices_stm(fens, device):
        expanded_fens = [expand_fen_string(fen) for fen in fens]
        emb_indices_list, stm_list = zip(*[NNUE.fen_to_indices_stm(fen, device) for fen in expanded_fens])
        emb_indices = torch.stack(emb_indices_list)
        # print('emb_indices', emb_indices.shape)
        stm = torch.stack(stm_list)
        # print('stm', stm.shape)
        return emb_indices, stm

    def forward_from_fens(self, fens, device):
        return self.forward(*self.fens_to_indices_stm(fens, device))

    def compute_white_win_prob_from_fen(self, fens, device):
        return NNUE.logit_to_white_win_prob(self.forward_from_fens(fens, device))

    def forward_from_fen(self, expanded_fen, device):
        return self.forward(*self.fen_to_indices_stm(expanded_fen, device))


def evaluate_prob_of_win(nnue, dataloader, dataset_name, loss_fn, global_step, summary_writer, device='cuda'):
    with torch.no_grad():
        total_test_samples = 0
        acc_loss = 0.0
        for test_batch in dataloader:
            test_batch = test_batch.to(device=device)
            pred_white_win_prob = nnue.compute_white_win_prob(test_batch.halfkp_indices, side_to_move=test_batch.side_to_move)
            total_test_samples += len(test_batch.mates)
            acc_loss += loss_fn(pred_white_win_prob, test_batch.white_win_prob_with_mates).sum().item()
        loss = acc_loss / total_test_samples
        print(
            f'{dataset_name} loss: {loss}, Total samples: {total_test_samples}')
        summary_writer.add_scalar(
            f'Loss/{dataset_name}',
            loss,
            global_step
        )
        print('Pred_win_prob:', pred_white_win_prob[:10])
        print('True_win_prob:', test_batch.white_win_prob_with_mates[:10])
        print('Mates:', test_batch.mates[:10])


# Function to save model weights as float32 in binary
def save_weights_binary(model, base_directory):
    assert not os.path.exists(base_directory)
    os.mkdir(base_directory)
    for name, param in model.named_parameters():
        filename = f"{base_directory}/{name}_{param.shape}.txt"
        param_data = param.data.cpu().numpy().astype('float32')
        param_data.tofile(filename, sep=' ')


def look_at_weights():
    device = 'cuda'
    nnue = NNUE(embedding_dim=1024, num_hidden1=8, num_hidden2=32).to(device=device)
    global_step = 240000
    if True:
        checkpoint_path = f'checkpoint_{global_step}.pth'
        checkpoint = torch.load(checkpoint_path)
        nnue.load_state_dict(checkpoint['nnue_state_dict'])
        print(checkpoint['nnue_state_dict'])
        for k, v in checkpoint['nnue_state_dict'].items():
            print(f'{k}: {v.shape}, {v.dtype}')
        global_step = checkpoint['global_step']
        save_weights_binary(nnue, base_directory=f'nnue_1_checkpoint_{global_step}')

def train_to_predict_prob_of_win():
    device = 'cuda'
    filepath = r"C:\Users\Ahmad-personal\PycharmProjects\chess_stackfish_evals\data\lichess_db_standard_rated_2024-02.jsonl"
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
    set_seed(60)
    eval_batch_size = 1024

    train_eval_dataloader = DataLoader(
        subset_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=1,
        collate_fn=nnue_collate_fn, pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=False
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True, num_workers=1, collate_fn=nnue_collate_fn,
        pin_memory=True, drop_last=True, prefetch_factor=2, persistent_workers=False
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=1, collate_fn=nnue_collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=False
    )

    nnue = NNUE(embedding_dim=2048, num_hidden1=16, num_hidden2=256).to(device=device)
    scaler = torch.cuda.amp.GradScaler()
    optim = Adam(nnue.parameters(), lr=0.001)

    global_step = 77500
    if True:
        checkpoint_path = f'checkpoint_{global_step}.pth'
        checkpoint = torch.load(checkpoint_path)
        nnue.load_state_dict(checkpoint['nnue_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']

    for param_group in optim.param_groups:
        param_group['lr'] = 0.001

    summary_writer = SummaryWriter('runs/nnue_8_big_2048_16_256')
    l2_loss_fn = torch.nn.MSELoss(reduction='none')

    # batch_iterator = skip_to_batch(train_dataloader, global_step)
    for batch in train_dataloader:
        global_step += 1
        optim.zero_grad()
        with autocast():
            batch = batch.to(device=device)

            pred_white_win_prob = nnue.compute_white_win_prob(batch.halfkp_indices, side_to_move=batch.side_to_move)
            # acc_loss += ((pred_evals - test_batch.cp_evals / 100.0) ** 2)[test_batch.cp_valid].sum().item()
            loss = l2_loss_fn(pred_white_win_prob, batch.white_win_prob_with_mates).mean()

        # Scales the loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        max_norm = 0.15
        scaler.unscale_(optim)  # Unscales the gradients of optimizer's assigned params in-place
        grad_norm = torch.nn.utils.clip_grad_norm_(nnue.parameters(), max_norm)

        # If gradients do not contain inf or NaN, updates the parameters
        scaler.step(optim)
        scaler.update()

        if global_step % 200 == 0:
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

        if global_step % 5000 == 0:
            nnue.eval()
            evaluate_prob_of_win(nnue, test_dataloader, 'Test-Set', l2_loss_fn, global_step, summary_writer)
            evaluate_prob_of_win(nnue, train_eval_dataloader, 'Train-Eval-Set', l2_loss_fn, global_step, summary_writer)
            nnue.train()

        if global_step % 2500 == 0:
            # Define the pattern for the checkpoint files
            checkpoint_pattern = "checkpoint_*.pth"

            # List all files matching the checkpoint pattern
            existing_checkpoints = glob.glob(checkpoint_pattern)

            # Save the new checkpoint
            checkpoint_path = f'checkpoint_{global_step}.pth'

            torch.save(
                {
                    'nnue_state_dict': nnue.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'global_step': global_step,
                }, checkpoint_path)
            print(f'Saved model at {checkpoint_path}.')

            # Delete all existing checkpoint files
            for checkpoint_file in existing_checkpoints:
                os.remove(checkpoint_file)
                print(f'Deleted {checkpoint_file}')


if __name__ == '__main__':
    # look_at_weights()
    train_to_predict_prob_of_win()
