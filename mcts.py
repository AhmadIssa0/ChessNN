
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
        return self.board.is_game_over()

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
        self.children = []
        self.total_value = 0
        self.visits = 0
        self.rollout_evaluator = rollout_evaluator
        self.evals = None  # will store pairs of (move, eval) from NN from the perspective of current player

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.total_value += result

    def uct_select_child(self):
        # untried_moves = [move for move in self.state.get_moves() if move not in [child.move for child in self.children]]
        # if untried_moves:
        #     move = random.choice(untried_moves)
        #     new_child = Node(move=move, parent=self, state=self.state.make_move(move), rollout_evaluator=self.rollout_evaluator)
        #     self.add_child(new_child)
        #     return new_child

        log_visits = math.log(self.visits)
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            # A win for a child means that we're losing!
            ucb1 = (1.0 - child.total_value) / (1.0 + child.visits) + 0.1 * math.sqrt(2 * log_visits / (1 + child.visits))
            if ucb1 > best_score:
                best_score = ucb1
                best_child = child
        return best_child

    def add_virtual_loss(self):
        current = self
        while current is not None:
            current.total_value += 1.0
            current.visits += 1
            current = current.parent

    def revert_virtual_loss(self):
        current = self
        while current is not None:
            current.total_value -= 1.0
            current.visits -= 1
            current = current.parent

    def is_terminal(self):
        return self.state.is_terminal()

    def expand(self):
        legal_moves = self.state.board.legal_moves
        evals = get_evals(self.state.board, legal_moves)
        self.evals = list(zip(legal_moves, evals))
        best_score = -float('inf')
        best_child = None
        for move, eval in self.evals:
            next_state = self.state.make_move(move)
            child_node = UCTNode(move=move, parent=self, state=next_state, rollout_evaluator=self.rollout_evaluator)
            child_node.total_value = 1.0 - eval
            self.add_child(child_node)
            if best_score < eval:
                best_score = eval
                best_child = child_node

        return best_child

    def rollout(self):
        return self.rollout_evaluator(self.state)

    def __str__(self, level=0):
        ret = "\t" * level
        if self.move is not None:
            ret += f"Move: {self.move}, Eval: {self.total_value / self.visits}, Visits: {self.visits}, Player after move: {self.state.player}\n"
        else:
            ret += f"Root Node, Wins: {self.total_value}, Visits: {self.visits}\n"

        for child in self.children:
            if level <= 0:
                ret += child.__str__(level + 1)
        return ret


def mcts(root, min_iterations, time_limit=2.0):
    start_time = time.time()
    i = 0
    while True:
        # Print progress every 100 iterations
        if i % 100 == 0:
            print('MCTS iteration:', i)

        # MCTS iteration process
        node = root
        while node.children:
            node = node.uct_select_child()
        if not node.is_terminal():
            node = node.expand()
        result = node.rollout()

        # Update nodes with the result
        while node is not None:
            node.update(result)
            node = node.parent
            result = 1.0 - result

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
    mcts(root, min_iterations=min_iterations)
    print(root)
    # Select the move with the highest number of visits
    best_move = max(root.children, key=lambda c: (c.visits, 1.0 - c.total_value / c.visits))
    return best_move.move, root.total_value / root.visits


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
while not state.is_terminal() and plies < 200:
    plies += 1
    min_iterations = 50 if plies < 100 else 200
    # min_iterations = 2000
    move, eval = mcts_move(state, min_iterations=min_iterations)
    print('Making move:', move, 'eval:', round(eval, 3), 'ply:', plies)
    state = state.make_move(move)
    print(state.board)
    if plies % 10 == 0:
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
