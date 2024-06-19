
import os

# Set the number of threads
# os.environ['OMP_NUM_THREADS'] = '12'

import random
import math
import chess
import torch
from main import ChessTranformer
import glob
import chess.pgn as pgn
import time
from typing import Optional, List
from bin_predictor import BinPredictor
import os
from torch.cuda.amp import autocast
import multiprocessing
from multiprocessing import Pool
from itertools import accumulate
from main import set_seed
from nnue import NNUE


class ChessState:
    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board

    def get_moves(self):
        return list(self.board.legal_moves)

    def make_move(self, move):
        new_board = self.board.copy()
        new_board.push(move)
        return ChessState(new_board)

    def is_terminal(self):
        return self.board.is_game_over(claim_draw=False) or self.board.is_repetition() or self.board.is_fifty_moves()

    def is_win(self, player_color):
        # Check for a win condition for the specified player color.
        # In chess, a win could be a checkmate.
        # This method needs to be adapted based on how you define a win for each player.
        if self.board.is_checkmate():
            # Assuming 'WHITE' and 'BLACK' are used to represent player colors.
            return True if (player_color == 'WHITE' and self.board.turn == chess.WHITE) or \
                           (player_color == 'BLACK' and self.board.turn == chess.BLACK) else False
        return False

    @property
    def player(self):
        return 'WHITE' if self.board.turn else 'BLACK'

class UCTNode:
    def __init__(self, device, move=None, move_idx=None, parent=None, state=None):
        self.move: chess.Move = move
        self.move_idx = move_idx  # index of the move in parent's legal_moves list
        self.parent: Optional[UCTNode] = parent
        self.state: ChessState = state
        self.legal_moves: List[chess.Move] = state.get_moves()
        self.is_expanded = False  # a node is expanded if the initial evals of its children have been set
        self.applied_virtual_loss_count = 0
        self.device = device
        if parent is None:
            self.plies_from_root = 0
        else:
            self.plies_from_root = parent.plies_from_root + 1

        self.children = {}  # Dict[move, UCTNode]
        # For the values below, we use absolute values in [0, 1], where '1' means white wins
        self.child_total_value = torch.zeros([len(self.legal_moves)], dtype=torch.float32, device=device)
        self.child_number_visits = torch.zeros([len(self.legal_moves)], dtype=torch.float32, device=device)

    @property
    def total_value(self):
        """Value of current node from white's perspective"""
        if self.is_root():
            best_move_idx = self.best_move_idx()
            return self.child_total_value[best_move_idx].item()
        return self.parent.child_total_value[self.move_idx].item()

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move_idx] = value

    @property
    def visits(self):
        if self.is_root():
            return self.child_number_visits.sum().item()
        return self.parent.child_number_visits[self.move_idx].item()

    @visits.setter
    def visits(self, value):
        self.parent.child_number_visits[self.move_idx] = value

    def is_root(self):
        return self.parent is None

    def add_child(self, child_node):
        self.children[child_node.move_idx] = child_node

    def best_move_idx(self):
        # best move is the one with the most visits, then the one with the best eval if there's a tie.
        max_visits = torch.max(self.child_number_visits).item()
        child_q = self.child_total_value / (1.0 + self.child_number_visits)
        if self.state.board.turn == chess.BLACK:
            child_q = 1.0 - child_q
        child_q = child_q * (self.child_number_visits == max_visits).float()
        return torch.argmax(child_q).item()

    def update(self, result):
        self.visits += 1
        self.total_value += result

    def uct_select_child(self):
        assert self.is_expanded

        log_visits = math.log(1 + self.visits)
        child_q = self.child_total_value / (1.0 + self.child_number_visits)
        if self.state.board.turn == chess.BLACK:
            child_q = 1.0 - child_q
        child_u = 0.04 * torch.sqrt(2 * log_visits / (1 + self.child_number_visits))
        # child_u = 0.05 * log_visits / (1 + self.child_number_visits)
        best_move_idx = torch.argmax(child_q + child_u).item()

        return self.maybe_add_child(best_move_idx)

    def eval_player_perspective(self):
        white_eval = self.total_value / (1 + self.visits)
        if self.state.board.turn == chess.BLACK:
            return 1.0 - white_eval
        return white_eval

    def eval_based_on_children(self):
        child_q = self.child_total_value / (1.0 + self.child_number_visits)
        if self.state.board.turn == chess.WHITE:
            return child_q.max().item()
        else:
            return child_q.min().item()

    def add_virtual_loss(self):
        current = self
        sgn = 1.0 if self.state.board.turn == chess.WHITE else -1.0
        while current is not None and current.parent is not None:  # parent is None for the root!
            if not current.is_root():
                current.total_value += sgn
                current.visits += 1
            current.applied_virtual_loss_count += 1
            current = current.parent
            sgn *= -1

    def revert_virtual_loss(self):
        current = self
        sgn = 1.0 if self.state.board.turn == chess.WHITE else -1.0
        while current is not None and current.parent is not None:
            if not current.is_root():
                current.total_value -= sgn
                current.visits -= 1
            current.applied_virtual_loss_count -= 1
            current = current.parent
            sgn *= -1

    def is_terminal(self):
        return self.state.is_terminal()

    def maybe_add_child(self, move_idx):
        if move_idx in self.children:
            return self.children[move_idx]
        else:
            move = self.legal_moves[move_idx]
            next_state = self.state.make_move(move)
            child_node = UCTNode(device=self.device, move=move, parent=self, state=next_state, move_idx=move_idx)
            self.add_child(child_node)
            return child_node

    def expand(self, child_evals):
        # child_evals is from the perspective of white
        self.is_expanded = True
        self.child_total_value = child_evals

    def terminal_state_eval(self):
        """From white's perspective."""
        board: chess.Board = self.state.board
        assert board.is_game_over(claim_draw=False) or board.is_repetition() or board.is_fifty_moves()
        if board.is_checkmate():  # current player got checkmated
            if board.turn == chess.WHITE:
                return -0.01 + self.plies_from_root * 1e-5
            else:
                return 1.01 - self.plies_from_root * 1e-5
        return 0.5

    def backup(self, result):
        """Back's up an eval from white's perspective."""
        current = self
        while current is not None and current.parent is not None:
            current.update(result)
            current = current.parent

    def __str__(self, level=0):
        ret = "\t" * level
        if not self.is_root():
            ret += f"Move: {self.move}, Abs-eval: {self.total_value / (1 + self.visits):.4f}, Visits: {self.visits}, Player after move: {self.state.player}\n"
        else:
            ret += f"Root Node, Abs-eval: {self.eval_based_on_children():.4f}, Visits: {self.visits}\n"

        for child in self.children.values():
            if level <= 0:
                ret += child.__str__(level + 1)
        return ret


class MCTSEngine:

    def __init__(self, root_dir=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stackfish_evals", device='cpu', model_type='transformer'):
        self.device = device
        self.model_type = model_type
        if model_type == 'transformer':
            transformer = ChessTranformer(
                bin_predictor=BinPredictor(), d_model=256, num_layers=4, nhead=4, dim_feedforward=4 * 256, norm_first=False
            # bin_predictor = BinPredictor(), d_model = 512, num_layers = 16, nhead = 8, dim_feedforward = 4 * 512
            ).to(device=device)
        elif model_type == 'nnue':
            transformer = NNUE(embedding_dim=1024, num_hidden1=8, num_hidden2=32).to(device=device)
        # root_dir = r"/mnt/c/Users/Ahmad-personal/PycharmProjects/chess_stackfish_evals"
        checkpoint_path = glob.glob(os.path.join(root_dir, "checkpoint_*.pth"))[0]
        checkpoint = torch.load(checkpoint_path)
        transformer.load_state_dict(checkpoint[f'{model_type}_state_dict'])
        transformer.eval()
        # transformer.transformer = torch.compile(transformer.transformer)
        self.transformer = transformer
        self.calls_to_eval = 0
        # The first time we call the transformer it's slow, so let's do it now not in a game!
        self.mcts_move(ChessState(chess.Board()), time_limit=2.0, verbose=False)
        print(f'Loaded transformer from checkpoint: {checkpoint_path}.')

    def get_evals_for_all_moves(self, nodes: List[UCTNode]):
        self.calls_to_eval += 1
        if len(nodes) == 0:
            return []
        node_separations = list(accumulate([len(node.legal_moves) for node in nodes]))[:-1]

        def _get_fens_and_draw_indices(node):
            node_fens = []
            node_draw_indices = []
            board: chess.Board = node.state.board
            for idx, move in enumerate(node.legal_moves):
                board.push(move)
                node_fens.append(board.fen())
                # if board.is_repetition() or board.is_fifty_moves() or board.is_stalemate():
                #     node_draw_indices.append(idx)
                board.pop()
            return node_fens, node_draw_indices

        fens_and_draw_indices = [_get_fens_and_draw_indices(node) for node in nodes]
        fens, draw_indices = list(zip(*fens_and_draw_indices))
        fens = sum(fens, [])
        draw_indices = sum(draw_indices, [])

        return self._compute_evals_from_fens(fens, node_separations, draw_indices)

    def _compute_evals_from_fens(self, fens, node_separations, draw_indices):
        with autocast():
            with torch.no_grad():
                white_evals = self.transformer.compute_white_win_prob_from_fen(fens, device=self.device)

                # white_evals, white_eval_stds = self.transformer.compute_bin_index_means_and_stds_from_fens(
                #     fens, device=self.device)
                # white_evals = white_evals / (self.transformer.bin_predictor.total_num_bins - 1.0)

                for idx in draw_indices:
                    white_evals[idx] = 0.5
                    # white_eval_stds[idx] = 0.0
                white_win_probs_split = torch.tensor_split(white_evals, node_separations)
                # white_eval_stds_split = torch.tensor_split(white_eval_stds, node_separations)
        return white_win_probs_split

    def mcts(self, root: UCTNode, node_batch_size, time_limit=2.0, verbose=True) -> None:
        start_time = time.time()
        i = 0
        while True:
            # MCTS iteration process
            nodes = []  # may have repeats
            for _ in range(node_batch_size):
                node = root
                while node.is_expanded:
                    node = node.uct_select_child()
                nodes.append(node)
                node.add_virtual_loss()

            nodes_set = set(nodes)
            terminal_nodes = list(node for node in nodes_set if node.is_terminal())
            non_terminal_nodes = list(node for node in nodes_set if not node.is_terminal())
            # print(get_evals_for_all_moves(non_terminal_nodes))
            for j, evals in enumerate(self.get_evals_for_all_moves(non_terminal_nodes)):
                node = non_terminal_nodes[j]
                # evals = evals.cpu()
                node.expand(evals)
                node.backup(result=node.eval_based_on_children())

            # Update nodes with the result
            for node in terminal_nodes:
                node.backup(result=node.terminal_state_eval())

            for node in nodes:
                node.revert_virtual_loss()

            i += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break
        if verbose:
            print(f"Completed {i} iterations in {elapsed_time:.2f} seconds.")

    def mcts_move(self, game: ChessState, node_batch_size=1, time_limit=2.0, verbose=True, root=None):
        """Continue's from root if specified, """
        root = UCTNode(device=self.device, state=game)
        with torch.no_grad():
            self.mcts(root, node_batch_size=node_batch_size, time_limit=time_limit, verbose=verbose)
        if verbose:
            print(root)

        best_move_idx = root.best_move_idx()
        best_move = root.legal_moves[best_move_idx]
        eval = root.child_total_value[best_move_idx] / (1 + root.child_number_visits[best_move_idx])
        return best_move, eval.item()

    def get_evals(self, board, moves):
        # evals are from current player's perspective
        fen_list = []
        for move in moves:
            board.push(move)
            fen_list.append(board.fen())
            board.pop()

        with torch.no_grad():
            evals = self.transformer.compute_avg_bin_index_from_fens(fen_list, device=self.device)
            evals = evals / (self.transformer.bin_predictor.total_num_bins - 1)
            evals = evals if board.turn == chess.WHITE else 1.0 - evals
            return evals.detach().cpu().tolist()

def run():
    # To use:
    # Initialize a chess board
    # fen = "3r2k1/3q1ppp/1p6/p2p4/1N2n3/4PQP1/5P1P/5RK1 w - - 0 30"
    # torch.jit.enable_onednn_fusion(True)
    # print(torch._dynamo.list_backends())

    set_seed(42)
    engine = MCTSEngine(model_type='nnue')
    board = chess.Board()
    state = ChessState(board)
    evals = [0.5]
    board.push(random.choice(list(board.legal_moves)))
    # evals = []
    plies = 0
    print(board)
    while not state.is_terminal() and plies < 10:
        plies += 1
        move, eval = engine.mcts_move(state, node_batch_size=1, time_limit=3)
        print('Making move:', move, 'eval:', round(eval, 3), 'ply:', plies)
        state = state.make_move(move)
        print(state.board)
        if plies % 5 == 0:
            print(pgn.Game.from_board(state.board))
        if state.board.turn == chess.WHITE:  # eval is originally from black's perspective
            eval = 1.0 - eval
        evals.append(eval)

    board = state.board

    print('Outcome:', board.outcome(claim_draw=True))

    game = pgn.Game().from_board(board)

    node = game
    for eval in evals:
        node = node.next()
        node.comment = f"[%wp {eval:.4f}]"

    # Export to PGN
    pgn_string = game.accept(pgn.StringExporter(headers=True, variations=True, comments=True))

    print(pgn_string)
    print('Calls:', engine.calls_to_eval)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('run()')
    run()

