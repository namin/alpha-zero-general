import Arena
from MCTS import MCTS
from _chess.ChessGame import ChessGame as Game
from _chess.ChessPlayers import HumanChessPlayer as HumanPlayer, RandomPlayer, StaticChessPlayer

from _chess.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play manually with the best temp agent.
"""

best_static = True
human_vs_cpu = False

g = Game()

hp = HumanPlayer(g).play
rp = RandomPlayer(g).play
sp = StaticChessPlayer(g).play

try:
  # nnet players
  n1 = NNet(g)
  n1.load_checkpoint('./temp/', 'best.pth.tar')
  args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
  mcts1 = MCTS(g, n1, args1)
  ap = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
except ValueError:
  print('warning: no AI found')
  ap = None

player1 = sp if best_static else ap
player2 = hp if human_vs_cpu else rp

arena = Arena.Arena(player1, player2, g, display=Game.display)

print(arena.playGames(2, verbose=True))
