import Arena
from MCTS import MCTS
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.TicTacToePlayers import HumanTicTacToePlayer as HumanPlayer
from tictactoe.keras.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play manually with the best temp agent.
"""

human_vs_cpu = True

g = Game()

hp = HumanPlayer(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp/', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
player1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

player2 = hp

arena = Arena.Arena(player1, player2, g, display=Game.display)

print(arena.playGames(2, verbose=True))
