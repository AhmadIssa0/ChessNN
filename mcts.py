
import random
import math
import chess
import torch
from main import ChessTranformer
import glob
import chess.pgn as pgn
import time
from typing import Optional


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
        return self.board.is_game_over(claim_draw=True)

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
    def __init__(self, move=None, parent=None, state=None, rollout_evaluator=None):
        self.move: chess.Move = move
        self.parent: Optional[UCTNode] = parent
        self.state: ChessState = state
        self.is_expanded = False
        self.children = []
        self.total_value = 0.0  # perspective of current player, positive means better
        self.visits = 0
        self.rollout_evaluator = rollout_evaluator
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
            ucb1 = (1.0 - child.total_value / (1.0 + child.visits)) + 0.1 * math.sqrt(2 * log_visits / (1 + child.visits))
            if ucb1 > best_score:
                best_score = ucb1
                best_child = child
        return best_child

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

    def expand(self, move_list, child_evals) -> None:
        # child_evals is from the perspective of white
        self.is_expanded = True
        for move, eval in zip(move_list, child_evals):
            next_state = self.state.make_move(move)
            child_node = UCTNode(move=move, parent=self, state=next_state, rollout_evaluator=self.rollout_evaluator)
            if next_state.board.turn == chess.WHITE:
                child_node.total_value = eval
            else:
                child_node.total_value = 1.0 - eval
            self.add_child(child_node)

    def rollout(self):
        return self.rollout_evaluator(self.state)

    def terminal_state_eval(self):
        # eval from perspective of current player
        board: chess.Board = self.state.board
        assert board.is_game_over(claim_draw=True)
        if board.is_checkmate():  # current player got checkmated
            return -9.0
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
            ret += f"Move: {self.move}, Eval: {self.total_value / (1 + self.visits)}, Visits: {self.visits}, Player after move: {self.state.player}, Virtual loss: {self.applied_virtual_loss_count}\n"
        else:
            ret += f"Root Node, Wins: {self.total_value}, Visits: {self.visits}\n"

        for child in self.children:
            if level <= 0:
                ret += child.__str__(level + 1)
        return ret


def get_evals_for_all_moves(nodes, device='cuda'):
    node_separations = []
    i = 0
    fens = []
    node_moves = []
    for node in nodes:
        board: chess.Board = node.state.board
        moves = []
        for move in board.legal_moves:
            moves.append(move)
            board.push(move)
            fens.append(board.fen())
            board.pop()
        node_moves.append(moves)
        i += len(moves)
        node_separations.append(i)
    if node_separations:
        node_separations = node_separations[:-1]
    with torch.no_grad():
        white_win_probs = transformer.compute_white_win_prob_from_fen(fens, device=device)
        white_win_probs_split = [elt.detach().cpu().tolist()
                                 for elt in torch.tensor_split(white_win_probs, node_separations)]
    return node_moves, white_win_probs_split


def mcts(root, node_batch_size, min_iterations, time_limit=2.0):
    start_time = time.time()
    i = 0
    while True:
        # print('iteration', i)
        # Print progress every 100 iterations
        if i % 100 == 0:
            print('MCTS iteration:', i)

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
        for j, (moves, evals) in enumerate(zip(*get_evals_for_all_moves(non_terminal_nodes))):
            node = non_terminal_nodes[j]
            node.expand(moves, evals)
            node.backup(result=max([1.0 - c.total_value for c in node.children]), result_from_white_perspective=False)

        # Update nodes with the result
        for node in terminal_nodes:
            node.backup(result=node.terminal_state_eval(), result_from_white_perspective=False)

        for node in nodes:
            node.revert_virtual_loss()

        i += 1

        # Check if it's time to stop
        elapsed_time = time.time() - start_time
        if i >= min_iterations and elapsed_time >= time_limit:
            break
        # elif i < min_iterations and elapsed_time >= time_limit:
        #     print("Warning: Time limit exceeded before reaching minimum iterations.")
        #     break

    print(f"Completed {i} iterations in {elapsed_time:.2f} seconds.")


def mcts_move(game, min_iterations=1000):
    root = UCTNode(state=game, rollout_evaluator=rollout_evaluator)
    mcts(root, node_batch_size=5, min_iterations=min_iterations)
    print(root)
    # Select the move with the highest number of visits
    best_move = max(root.children, key=lambda c: (c.visits, 1.0 - c.total_value / (1 + c.visits)))
    return best_move.move, root.total_value / (1 + root.visits)


def piece_value(piece):
    if piece is None:
        return 0
    value = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }[piece.piece_type]
    return value if piece.color == chess.WHITE else -value


def board_value(board):
    total_value = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        total_value += piece_value(piece)
    return total_value


device = 'cuda'
transformer = ChessTranformer(d_model=512, num_layers=8, nhead=8, dim_feedforward=4*512).to(device=device)
checkpoint_path = glob.glob("checkpoint_*.pth")[0]
checkpoint = torch.load(checkpoint_path)
transformer.load_state_dict(checkpoint['transformer_state_dict'])
transformer.eval()


def get_evals(board, moves):
    # evals are from current player's perspective
    fen_list = []
    for move in moves:
        board.push(move)
        fen_list.append(board.fen())
        board.pop()

    with torch.no_grad():
        evals = transformer.compute_white_win_prob_from_fen(fen_list, device=device)
        evals = evals if board.turn == chess.WHITE else 1.0 - evals
        return evals.detach().cpu().tolist()


def rollout_evaluator(chess_state):
    board: chess.Board = chess_state.board
    if board.is_game_over(claim_draw=True):
        if chess_state.is_terminal():
            return -9.0 if chess_state.is_win(chess_state.player) else 10.0  # player is who's moving AFTER the mate
        if board.can_claim_draw():
            return 0.5

    with torch.no_grad():
        white_win_prob = transformer.compute_white_win_prob_from_fen(
            [chess_state.board.fen()], device=device)[0].item()
        piece_value = (board_value(chess_state.board) / 39.0 + 1.0) / 2.0  # between 0 and 1

        eval = white_win_prob + (max(0.0, white_win_prob - 0.95) + max(0.0, 0.05 - white_win_prob)) * 10 * piece_value
        return eval if chess_state.board.turn == chess.WHITE else 1.0 - eval
        # return white_win_prob if chess_state.board.turn == chess.WHITE else 1.0 - white_win_prob


# To use:
# Initialize a chess board
# fen = "3r2k1/3q1ppp/1p6/p2p4/1N2n3/4PQP1/5P1P/5RK1 w - - 0 30"
board = chess.Board()
state = ChessState(board)
evals = [0.5]
board.push(random.choice(list(board.legal_moves)))
# evals = []
plies = 0
print(board)
while not state.is_terminal() and plies < 200:
    plies += 1
    min_iterations = 50
    # min_iterations = 2000
    move, eval = mcts_move(state, min_iterations=min_iterations)
    print('Making move:', move, 'eval:', round(eval, 3), 'ply:', plies)
    state = state.make_move(move)
    print(state.board)
    if plies % 5 == 0:
        print(pgn.Game.from_board(state.board))
    if state.board.turn == chess.WHITE:  # eval is originally from black's perspective
        eval = 1.0 - eval
    evals.append(eval)

board = state.board
print('Outcome:', board.outcome())

game = pgn.Game().from_board(board)
node = game
for eval in evals:
    node = node.next()
    node.comment = f"[%wp {eval:.4f}]"

# Export to PGN
pgn_string = game.accept(pgn.StringExporter(headers=True, variations=True, comments=True))

print(pgn_string)
