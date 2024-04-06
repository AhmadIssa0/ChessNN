
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
    def __init__(self, move=None, parent=None, state=None):
        self.move: chess.Move = move
        self.parent: Optional[UCTNode] = parent
        self.state: ChessState = state
        self.is_expanded = False
        self.children = []
        self.total_value = 0.0  # perspective of current player, positive means better
        self.init_eval_std = 0.0  # initial std of eval according to bin predictor
        self.visits = 0
        self.applied_virtual_loss_count = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.total_value += result

    def uct_select_child(self):
        assert self.is_expanded

        log_visits = math.log(self.visits)
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            # A win for a child means that we're losing!
            ucb1 = (1.0 - child.total_value / (1.0 + child.visits)) + self.init_eval_std * math.sqrt(2 * log_visits / (1 + child.visits))
            if ucb1 > best_score:
                best_score = ucb1
                best_child = child
        return best_child

    def eval_player_perspective(self):
        return self.total_value / (1 + self.visits)

    def add_virtual_loss(self):
        current = self
        while current is not None:
            current.total_value += 1.0
            current.visits += 1
            current.applied_virtual_loss_count += 1
            current = current.parent

    def revert_virtual_loss(self):
        current = self
        while current is not None:
            current.total_value -= 1.0
            current.visits -= 1
            current.applied_virtual_loss_count -= 1
            current = current.parent

    def is_terminal(self):
        return self.state.is_terminal()

    def expand(self, move_list, child_evals, child_stds) -> None:
        # child_evals is from the perspective of white
        self.is_expanded = True
        for move, eval, eval_std in zip(move_list, child_evals, child_stds):
            next_state = self.state.make_move(move)
            child_node = UCTNode(move=move, parent=self, state=next_state)
            child_node.init_eval_std = eval_std
            if next_state.board.turn == chess.WHITE:
                child_node.total_value = eval
            else:
                child_node.total_value = 1.0 - eval
            self.add_child(child_node)

    def terminal_state_eval(self):
        # eval from perspective of current player
        board: chess.Board = self.state.board
        assert board.is_game_over(claim_draw=True)
        if board.is_checkmate():  # current player got checkmated
            return 0.0
        return 0.5

    def backup(self, result, result_from_white_perspective):
        current = self
        if result_from_white_perspective and self.state.board.turn != chess.WHITE:
            result = 1.0 - result
        while current is not None:
            current.update(result)
            current = current.parent
            result = 1.0 - result

    def __str__(self, level=0):
        ret = "\t" * level
        if self.move is not None:
            ret += f"Move: {self.move}, Eval: {self.total_value / (1 + self.visits):.4f}, Init-std: {self.init_eval_std:.4f}, Visits: {self.visits}, Player after move: {self.state.player}\n"
        else:
            ret += f"Root Node, Wins: {self.total_value}, Visits: {self.visits}\n"

        for child in self.children:
            if level <= 0:
                ret += child.__str__(level + 1)
        return ret


class MCTSEngine:

    def __init__(self, root_dir=r"C:\Users\Ahmad-personal\PycharmProjects\chess_stackfish_evals", device='cuda'):
        self.device = device
        transformer = ChessTranformer(
            bin_predictor=BinPredictor(), d_model=512, num_layers=8, nhead=8, dim_feedforward=4 * 512
        ).to(device=device)
        checkpoint_path = glob.glob(os.path.join(root_dir, "checkpoint_*.pth"))[0]
        checkpoint = torch.load(checkpoint_path)
        transformer.load_state_dict(checkpoint['transformer_state_dict'])
        transformer.eval()
        transformer.transformer = torch.compile(transformer.transformer)
        self.transformer = transformer
        # The first time we call the transformer it's slow, so let's do it now not in a game!
        self.mcts_move(ChessState(chess.Board()), time_limit=1.0, verbose=False)
        print(f'Loaded transformer from checkpoint: {checkpoint_path}.')

    def get_evals_for_all_moves(self, nodes: List[UCTNode]):
        if len(nodes) == 0:
            return [], []
        node_separations = []
        i = 0
        fens = []
        node_moves = []
        draw_indices = []
        for node in nodes:
            board: chess.Board = node.state.board
            moves = []
            for idx, move in enumerate(board.legal_moves):
                moves.append(move)
                board.push(move)
                fens.append(board.fen())
                if board.can_claim_draw():
                    draw_indices.append(idx)
                board.pop()
            node_moves.append(moves)
            i += len(moves)
            node_separations.append(i)
        if node_separations:
            node_separations = node_separations[:-1]
        with torch.no_grad():
            # white_evals = transformer.compute_white_win_prob_from_fen(fens, device=device)
            # white_evals = self.transformer.compute_avg_bin_index_from_fens(fens, device=self.device)
            white_evals, white_eval_stds = self.transformer.compute_bin_index_means_and_stds_from_fens(
                fens, device=self.device)
            white_evals = white_evals / (self.transformer.bin_predictor.total_num_bins - 1.0)
            white_eval_stds = white_eval_stds / (self.transformer.bin_predictor.total_num_bins - 1.0)
            for idx in draw_indices:
                white_evals[idx] = 0.5
                white_eval_stds[idx] = 0.0
            white_win_probs_split = [elt.detach().cpu().tolist()
                                     for elt in torch.tensor_split(white_evals, node_separations)]
            white_eval_stds_split = [elt.detach().cpu().tolist()
                                     for elt in torch.tensor_split(white_eval_stds, node_separations)]
        return node_moves, white_win_probs_split, white_eval_stds_split

    def mcts(self, root: UCTNode, node_batch_size, time_limit=2.0, verbose=True) -> None:
        start_time = time.time()
        i = 0
        while True:
            # MCTS iteration process
            nodes = []  # may have repeats
            for _ in range(node_batch_size):
                node = root
                while node.children:
                    node = node.uct_select_child()
                nodes.append(node)
                node.add_virtual_loss()

            nodes_set = set(nodes)
            terminal_nodes = list(node for node in nodes_set if node.is_terminal())
            non_terminal_nodes = list(node for node in nodes_set if not node.is_terminal())
            # print(get_evals_for_all_moves(non_terminal_nodes))
            for j, (moves, evals, eval_stds) in enumerate(zip(*self.get_evals_for_all_moves(non_terminal_nodes))):
                node = non_terminal_nodes[j]
                node.expand(moves, evals, eval_stds)
                scores = [1.0 - c.total_value for c in node.children]
                assert len(scores) > 0, 'Cant backup when there are no children nodes!'
                node.backup(result=max(scores), result_from_white_perspective=False)

            # Update nodes with the result
            for node in terminal_nodes:
                node.backup(result=node.terminal_state_eval(), result_from_white_perspective=False)

            for node in nodes:
                node.revert_virtual_loss()

            i += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_limit:
                break
        if verbose:
            print(f"Completed {i} iterations in {elapsed_time:.2f} seconds.")

    def mcts_move(self, game: ChessState, node_batch_size=5, time_limit=2.0, verbose=True, root=None):
        """Continue's from root if specified, """
        root = UCTNode(state=game)
        self.mcts(root, node_batch_size=node_batch_size, time_limit=time_limit, verbose=verbose)
        if verbose:
            print(root)

        if root.children:
            # Select the move with the highest number of visits
            best_child = max(root.children, key=lambda c: (c.visits, 1.0 - c.total_value / (1 + c.visits)))
            best_move = best_child.move
        else:
            # root is a terminal state, if it's a draw that's being claimed then randomly pick a move,
            # we'll claim draw anyway so it doesn't matter
            legal_moves = game.get_moves()
            best_move = legal_moves[0] if legal_moves else None

        return best_move, root.total_value / (1 + root.visits)

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
    engine = MCTSEngine()
    board = chess.Board()
    state = ChessState(board)
    evals = [0.5]
    board.push(random.choice(list(board.legal_moves)))
    # evals = []
    plies = 0
    print(board)
    while not state.is_terminal() and plies < 10:
        plies += 1
        move, eval = engine.mcts_move(state, time_limit=3.0)
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

if __name__ == '__main__':
    import cProfile
    # cProfile.run('run()')
    run()

