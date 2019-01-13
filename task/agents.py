# @Author: VoldikSS
# @Date: 2019-01-13 15:56:18
# @Last Modified by: voldikss
# @Last Modified time: 2019-01-13 23:30:49

import sys
sys.path.append(".")

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model_0_1024.h5")

from game2048.agents import Agent
from .util import board_to_onehot
import numpy as np

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


class CNNAgent(Agent):
    '''Agent using convolutional neural network to play the game.'''

    def __init__(self, game, display=None):
        super().__init__(game, display)
        from keras.models import load_model
        self.model = load_model(model_path)

    def step(self):
        board = np.array([board_to_onehot(self.game.board)])
        direction = int(self.model.predict(board).argmax())
        return direction
