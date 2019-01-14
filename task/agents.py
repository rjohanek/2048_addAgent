# @Author: VoldikSS
# @Date: 2019-01-13 15:56:18
# @Last Modified by: voldikss
# @Last Modified time: 2019-01-14 09:08:24

import sys
sys.path.append(".")

import os
from game2048.agents import Agent
from .util import board_to_onehot
import numpy as np
from keras.models import load_model


class PlanningAgent(Agent):
    '''Agent using tree-search to play the game.'''

    def __init__(self, game, display=None):
        super().__init__(game, display)
        from .planning import board_to_move
        self.search_func = board_to_move
        # map_depth, can be adjusted
        self.max_depth = 4

    def step(self):
        direction = self.search_func(self.game.board, self.max_depth)
        return direction


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model_0_1024.h5")
my_model = load_model(model_path)


class CNNAgent(Agent):
    '''Agent using convolutional neural network to play the game.'''

    def __init__(self, game, display=None):
        super().__init__(game, display)

    def step(self):
        board = np.array([board_to_onehot(self.game.board)])
        direction = int(my_model.predict(board).argmax())
        return direction
