

import chess
import torch
from main import ChessTranformer
import chess.pgn as pgn
import random
from bin_predictor import BinPredictor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
device = 'cuda'

bin_pred = BinPredictor()


def best_move_from_fen(transformer, board):
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
        evals = transformer.compute_avg_bin_index_from_fens(fen_list, device=device)
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

    transformer = ChessTranformer(
        bin_predictor=BinPredictor(), d_model=512, num_layers=8, nhead=8, dim_feedforward=4*512).to(device=device)
    import glob

    checkpoint_path = glob.glob("checkpoint_*.pth")[0]
    checkpoint = torch.load(checkpoint_path)
    transformer.load_state_dict(checkpoint['transformer_state_dict'])
    transformer.eval()


if __name__ == '__main__':
    run()