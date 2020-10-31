from MCTS import MCTS
from chess.ChessGame import ChessGame as Game
from chess.pytorch.NNet import NNetWrapper as NNet
from chess.pytorch.NNet import args
from chess.ChessGame import to_np, to_move, mirror_move, who

import numpy as np
from utils import *

args.cuda = False
g = Game()
n1 = NNet(g)
n1.load_checkpoint('./temp/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

def alpha_zero_strategy(board):
    move = to_move(player1(g.getCanonicalForm(board, who(board.turn))))
    if not board.turn:
        move = mirror_move(move)
    return move
