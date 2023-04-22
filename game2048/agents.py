import numpy as np
import collections


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
            
                if self.display is not None:
                    self.display.display(self.game)

    def play_learn(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        state_pairs = []

        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            state = self.game.board
            self.game.move(direction)
            new_state = self.game.board
            n_iter += 1
            state_pairs.append((state, new_state))

            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
            
                if self.display is not None:
                    self.display.display(self.game)
        counter = collections.Counter(state_pairs)
        print(counter)
        return counter

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
