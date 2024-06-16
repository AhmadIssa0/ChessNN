
import chess
import torch
from main import ChessTranformer
import chess.pgn as pgn
import random
from bin_predictor import BinPredictor
from torch.cuda.amp import autocast
from nnue import NNUE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
device = 'cpu'

bin_pred = BinPredictor()


def best_move_from_fen(model, board, model_type):
    legal_moves = list(board.legal_moves)
    fen_list = []
    three_fold_indices = []
    for i, move in enumerate(legal_moves):
        board.push(move)
        fen_list.append(board.fen())
        if board.can_claim_draw() or board.is_stalemate():
            three_fold_indices.append(i)
        board.pop()

    with torch.no_grad():
        # evals = transformer.compute_white_win_prob_from_fen(fen_list, device=device)
        if model_type == 'transformer':
            evals = model.compute_avg_bin_index_from_fens(fen_list, device=device)
        elif model_type == 'nnue':
            evals = model.forward_from_fens(fen_list, device=device)

        for i in three_fold_indices:
            if board.turn:
                evals[i] = min((bin_pred.total_num_bins + 1) / 2, evals[i])
            else:
                evals[i] = max((bin_pred.total_num_bins + 1) / 2, evals[i])
    if board.turn:
        best_move_idx = evals.argmax().item()
    else:
        best_move_idx = evals.argmin().item()
    best_move = legal_moves[best_move_idx]
    print('All evals:', list(zip(evals, legal_moves)))
    return best_move, evals[best_move_idx].item()


def run():
    # Initialize a board from a FEN string
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
    # fen = "6k1/1pp2R2/2n3P1/p3p1qP/1P5b/P1P5/2K1R3/8 b - - 6 56"
    board = chess.Board(fen)

    model_type = 'nnue'
    if model_type == 'transformer':
        chess_network = ChessTranformer(
            bin_predictor=BinPredictor(), d_model=256, num_layers=4, nhead=4, dim_feedforward=4*256, norm_first=False).to(device=device)
    elif model_type == 'nnue':
        chess_network = NNUE(embedding_dim=1024, num_hidden1=8, num_hidden2=32).to(device=device)
    import glob

    checkpoint_path = glob.glob("checkpoint_*.pth")[0]
    checkpoint = torch.load(checkpoint_path)
    chess_network.load_state_dict(checkpoint[f'{model_type}_state_dict'])
    chess_network.eval()

    evals = [0.5, 0.5]
    stds = [0.0, 0.0]
    board.push(random.choice(list(board.legal_moves)))
    board.push(random.choice(list(board.legal_moves)))
    ply = 2

    while not board.is_game_over():
        # with torch.no_grad():
        #     probs = transformer.compute_bin_probabilities_from_fens([board.fen()], device)[0].tolist()
        #     indices = range(len(probs))
        #     plt.bar(indices, probs)
        #     plt.savefig(f'plots/probabilities_{ply}.png')
        #     plt.close()
        #     ply += 1
        best_move, eval = best_move_from_fen(chess_network, board, model_type)
        print('Making move:', best_move, 'eval after making move:', eval)
        print(board)
        board.push(best_move)
        evals.append(2.0 * eval / (bin_pred.total_num_bins - 1) - 1.0)
        # with torch.no_grad():
        #     index_means, index_stds = chess_network.compute_bin_index_means_and_stds_from_fens([board.fen()], device)
            # stds.append(index_stds[0].item() / (bin_pred.total_num_bins - 1))

    print('Outcome:', board.outcome())
    # print(pgn.Game.from_board(board))

    game = pgn.Game().from_board(board)
    node = game
    for i, eval in enumerate(evals):
        node = node.next()
        node.comment = f"[%eval {eval:.4f}]"

    # Export to PGN
    pgn_string = game.accept(pgn.StringExporter(headers=True, variations=True, comments=True))

    print(pgn_string)


if __name__ == '__main__':
    with autocast():
        run()

