
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

evals = [0.5, 0.5]
stds = [0.0, 0.0]
board.push(random.choice(list(board.legal_moves)))
board.push(random.choice(list(board.legal_moves)))
ply = 2

while not board.is_game_over():
    with torch.no_grad():
        probs = transformer.compute_bin_probabilities_from_fens([board.fen()], device)[0].tolist()
        indices = range(len(probs))
        plt.bar(indices, probs)
        plt.savefig(f'plots/probabilities_{ply}.png')
        plt.close()
        ply += 1
    best_move, eval = best_move_from_fen(transformer, board)
    print('Making move:', best_move, 'eval after making move:', eval)
    print(board)
    board.push(best_move)
    evals.append(2.0 * eval / (bin_pred.total_num_bins - 1) - 1.0)
    with torch.no_grad():
        index_means, index_stds = transformer.compute_bin_index_means_and_stds_from_fens([board.fen()], device)
        stds.append(index_stds[0].item() / (bin_pred.total_num_bins - 1))

print('Outcome:', board.outcome())
# print(pgn.Game.from_board(board))

game = pgn.Game().from_board(board)
node = game
for i, eval in enumerate(evals):
    node = node.next()
    node.comment = f"[%eval {eval:.4f} +/- {stds[i]}]"

# Export to PGN
pgn_string = game.accept(pgn.StringExporter(headers=True, variations=True, comments=True))

print(pgn_string)

# # Print the board
# print(board)
#
# # Generate all legal moves from the position
# legal_moves = list(board.legal_moves)
# print("Legal moves:", legal_moves)
#
# # Make a move
# move = legal_moves[0]  # Choose the first legal move for demonstration
# board.push(move)  # Make the move
#
# # Print the updated board
# print("After making a move:")
# print(board)
# print(board.fen())
#
