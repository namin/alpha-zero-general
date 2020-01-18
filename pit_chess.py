import Arena
from MCTS import MCTS
from chess.ChessGame import ChessGame as Game
from chess.ChessPlayers import HumanChessPlayer as HumanPlayer, RandomChessPlayer as RandomPlayer, StaticChessPlayer

from chess.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

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

player1 = sp
player2 = rp

arena = Arena.Arena(player1, player2, g, display=Game.display)

print(arena.playGames(10, verbose=False))
