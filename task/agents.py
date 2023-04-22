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
        from .planning import predict
        self.search_func = predict
        # map_depth, can be adjusted
        self.max_depth = 4

    def step(self):
        results = self.search_func(self.game.board, self.max_depth)
        direction = max(results, key=lambda x: x[1])[0]
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


class MarkovAgent(Agent):
    '''Agent using Markov Decision Process, using Value Iteration'''

    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.model = MarkovModel(game)

    def step(self):
        direction = np.random.randint(0, 4)
        return direction

# Markov Decision Process
# only implemented for games 2x2
class MarkovModel():

    def __init__(self):
        # As established in game.py 0=left, 1=down, 2=right, 3=up
        self.actions = [0, 1, 2, 3]

        # even with a board size of 2x2 there are combinatorily 529 states
        # so instead only starting states are listed here, determined by the procedure in game.py
        # where up to two tiles may be generated with values 2 or 4
        # the rest of the states are enumerated in a DFS manner as moves are made, see add_states()
        self.states = self.generate_starting_states()

        # rewards only for terminal states +10 for win -10 for loss
        # so reward is 0 for the first 32 starting states
        self.rewards = [0] * 32

        # discounting value
        self.discount = 0.95

        # Load probabilities that were learned during exploration
        self.probability = []  # load_probabilities()

    def generate_starting_states():
        return [
            # one 2 generated (4 options)
            [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0],
            # one 4 generated (4 options)
            [0, 0, 0, 2], [0, 0, 2, 0], [0, 2, 0, 0], [2, 0, 0, 0],
            # two 2 tiles generated (6 options)
            [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0], [
                0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]
            # two 4 tiles generated (6 options)
            [0, 0, 2, 2], [2, 2, 0, 0], [0, 2, 2, 0], [
                    0, 2, 0, 2], [2, 0, 2, 0], [2, 0, 0, 2]
            # one 2 and one 4 generated (12 options)
            [0, 0, 2, 1], [0, 2, 0, 1], [0, 0, 2, 1],
            [0, 0, 1, 2], [2, 0, 1, 0], [0, 2, 1, 0],
            [0, 1, 0, 2], [0, 1, 2, 0], [0, 1, 0, 2],
            [1, 2, 0, 0], [1, 0, 2, 0], [1, 0, 0, 2]
        ]

    def add_states(self, state):
        NotImplementedError()

    def value_iteration(self):
        NotImplementedError()

    def policy_iteration(self):
        NotImplementedError()

    def load_probabilities(self):
        # chose an exploration policy
        # estimate T (slide 21 lecture 11)
        NotImplementedError()
