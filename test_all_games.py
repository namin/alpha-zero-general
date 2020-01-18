""""

    This is a Regression Test Suite to automatically test all combinations of games and ML frameworks. Each test
    plays two quick games using an untrained neural network (randomly initialized) against a random player.

    In order for the entire test suite to run successfully, all the required libraries must be installed.  They are:
    Pytorch, Keras, Tensorflow.

     [ Games ]      Pytorch     Tensorflow  Keras
      -----------   -------     ----------  -----
    - Othello       [Yes]       [Yes]       [Yes]
    - TicTacToe                             [Yes]
    - Connect4                  [Yes]
    - Gobang                    [Yes]       [Yes]
    - Chess         [Yes]

"""

import unittest

import Arena
from MCTS import MCTS

from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as OthelloPytorchNNet
from othello.tensorflow.NNet import NNetWrapper as OthelloTensorflowNNet
from othello.keras.NNet import NNetWrapper as OthelloKerasNNet

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.TicTacToePlayers import *
from tictactoe.keras.NNet import NNetWrapper as TicTacToeKerasNNet

from connect4.Connect4Game import Connect4Game
from connect4.Connect4Players import *
from connect4.tensorflow.NNet import NNetWrapper as Connect4TensorflowNNet

from gobang.GobangGame import GobangGame
from gobang.GobangPlayers import *
from gobang.keras.NNet import NNetWrapper as GobangKerasNNet
from gobang.tensorflow.NNet import NNetWrapper as GobangTensorflowNNet

from chess.ChessGame import ChessGame
from chess.ChessPlayers import *
from chess.pytorch.NNet import NNetWrapper as ChessPytorchNNet

import numpy as np
from utils import *

class TestAllGames(unittest.TestCase):

    @staticmethod
    def execute_game_test(game, neural_net):
        rp = RandomPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, rp, game)
        print(arena.playGames(2, verbose=False))

    def test_othello_pytorch(self):
        self.execute_game_test(OthelloGame(6), OthelloPytorchNNet)

    def test_othello_tensorflow(self):
        self.execute_game_test(OthelloGame(6), OthelloTensorflowNNet)

    def test_othello_keras(self):
        self.execute_game_test(OthelloGame(6), OthelloKerasNNet)

    def test_tictactoe_keras(self):
        self.execute_game_test(TicTacToeGame(), TicTacToeKerasNNet)

    def test_connect4_tensorflow(self):
        self.execute_game_test(Connect4Game(), Connect4TensorflowNNet)

    def test_gobang_keras(self):
        self.execute_game_test(GobangGame(), GobangKerasNNet)

    def test_gobang_tensorflow(self):
        self.execute_game_test(GobangGame(), GobangTensorflowNNet)

    def test_chess_pytorch(self):
        self.execute_game_test(ChessGame(), ChessPytorchNNet)


if __name__ == '__main__':
    unittest.main()
