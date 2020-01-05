import chess
import random
import numpy as np
from _chess.ChessGame import who, from_move, mirror_move

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, who(board.turn))
        moves = np.argwhere(valids==1)
        return random.choice(moves)[0]

def move_from_uci(board, uci):
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        print('expected an UCI move')
        return None
    if move not in board.legal_moves:
        print('expected a valid move')
        return None
    return move

class HumanChessPlayer():
    def __init__(self, game):
        pass

    def play(self, board):
        human_move = input()
        mboard = board
        if board.turn:
            mboard = board.mirror()
        move = move_from_uci(mboard, human_move.strip())
        if move is None:
            print('try again, e.g., %s' % random.choice(list(mboard.legal_moves)).uci())
            return self.play(board)
        if board.turn:
            move = mirror_move(move)
        return from_move(move)

class StrategicChessPlayer():
    def __init__(self, game, strategy):
        self.strategy = strategy

    def play(self, board):
        mboard = board
        if board.turn:
            mboard = board.mirror()
        move = self.strategy(mboard)
        if board.turn:
            move = mirror_move(move)
        return from_move(move)

MIN_SCORE = -1000
MAX_SCORE = -MIN_SCORE

piece_values = {chess.KING: 0,
                chess.PAWN: 1,
                chess.BISHOP: 3,
                chess.KNIGHT: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9}

def evaluate_board(board):
    score = 0
    for piece in board.piece_map().values():
        value = piece_values[piece.piece_type]
        factor = 1 if piece.color == board.turn else -1
        score += factor * value
    score -= 100 if board.is_checkmate() else 0
    return score

def evaluate_move(move, board, evaluate_board):
    board.push(move)
    score = -evaluate_board(board)
    board.pop()
    return score

def evaluation_strategy(board, evaluate_move):
    best_moves = []
    best_score = MIN_SCORE
    for move in board.legal_moves:
        score = evaluate_move(move, board)
        if score > best_score:
           best_score = score
           best_moves = [move]
        elif score == best_score:
           best_moves.append(move)
    return random.choice(best_moves)

def static_evaluate_move(move, board):
    return evaluate_move(move, board, evaluate_board)

def static_strategy(board):
    return evaluation_strategy(board, static_evaluate_move)

class StaticChessPlayer(StrategicChessPlayer):
    def __init__(self, game):
        super().__init__(game, static_strategy)
