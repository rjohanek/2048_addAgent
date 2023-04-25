# @Author: VoldikSS
# @Date: 2019-01-13 15:56:18
# @Last Modified by: voldikss
# @Last Modified time: 2019-01-14 09:08:24

import ast
import json
from keras.models import load_model
import numpy as np
from .util import board_to_onehot
from game2048.agents import Agent, convert_state
import os
import sys

sys.path.append(".")


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


class GreedyAgent(Agent):
    '''Agent using tree-search to play the game.'''

    def __init__(self, game, display=None):
        super().__init__(game, display)
        from .planning import merge
        self.merge = merge

    def step(self):
        results = []
        for dir in range(4):
            # this is how the game.py file calculates score of a game/board
            results.append((dir, int(self.merge(self.game.board, dir).max())))
        direction = max(results, key=lambda x: x[1])[0]
        return direction


class LearningAgent(Agent):
    '''Agent learning probabilities of state transitions using a combination the pre-existing planning agent 90% of the time
       and random selection the other 10%.'''

    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.greedyAgent = GreedyAgent(game)
        self.counter = 0

    def step(self):
        if self.counter / 10 == 0:
            direction = np.random.randint(0, 4)
            self.counter += 1
        else:
            direction = self.greedyAgent.step()
            self.counter += 1
        return direction


class MarkovAgent(Agent):
    '''Agent using Markov Decision Process, using Value Iteration'''

    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.model = MarkovModel(game)

    def step(self):
        direction = self.model.policies[convert_state(self.game.board)]
        return direction


# Markov Decision Process
# only implemented for games 2x2
# initialized with states and transitions from previous exploration

class MarkovModel():

    def __init__(self, game, discount=0.85):
        # As established in game.py 0=left, 1=down, 2=right, 3=up
        self.actions = [0, 1, 2, 3]

        # load in values learned by learning agent during exploration
        loaded_data = load_states_probs()

        # includes all states seen in 100 games played on a 2x2 board up to score 32
        self.states = loaded_data["states"]

        # Load transition probabilities that were learned during exploration
        self.transitions = ast.literal_eval(loaded_data["probs"])

        # rewards only for terminal states +10 for win -10 for loss, 0 for all other
        self.rewards = determine_rewards(self.states, game.score_to_win)

        # discounting value
        self.discount = discount

        # Set value iteration parameters
        self.k = 100  # Maximum number of iterations
        self.delta = 1e-400  # Error tolerance
        self.values = [0]*len(self.states)  # Initialize values
        self.policies = [None]*len(self.states)  # Initialize policy

        self.value_iteration()

    def value_iteration(self):
        # starting at 0, do value iteration for k time, default 100
        for i in range(self.k):
            update_diff = 0 
            updated_values = [0]*len(self.states)  # Initialize values

            # go through every state
            for i in range(len(self.states)):
                max_val = 0
                s = self.states[i]
                for a in self.actions:

                    # Compute the state value
                    val = self.rewards[i]  # Get reward value of state
                    
                    # go through all states again
                    # could be improved by only going through states that are known successors
                    for j in range(len(self.states)):
                        s_next = self.states[j]
                        # if there is a non zero probability
                        if (s, s_next, a) in self.transitions:
                            # Add discounted downstream values
                            val += self.transitions[(s, s_next, a)] * (self.discount * self.values[j])

                    # Store best value so far
                    if val > max_val:
                        max_val = val

                    # Update policy based on new values
                    if self.values[i] < val:
                        # Store action with highest value
                        self.policies[i] = self.actions[a]

                updated_values[i] = max_val 

                # Update difference
                update_diff = max(update_diff, abs(self.values[i] - updated_values[i]))

            # Update value functions
            self.values = updated_values

            # convergence
            if update_diff < self.delta:
                break

    def policy_iteration(self):
        NotImplementedError()


def load_states_probs():
    # estimate T and S through learning agent that uses greedy 90% and random 10%
    # slight modifications were made prior to loading in:
    # convert ' to " around the words states and probs
    # convert ' to " everywhere in states list
    # add " " around the entire value of probs
    with open("learned_states_probs.txt", "r") as f:
        data = f.read()
    dict = json.loads(data, strict=False)
    return dict


def determine_rewards(states, score_to_win):
    rewards = []
    for s in states:
        if is_win(s, score_to_win):
            rewards.append(10)
        elif is_loss(s):
            rewards.append(-10)
        else:
            rewards.append(0)
    return rewards


def is_win(state, score_to_win):
    state_values = state.split(".")
    result = False
    for val in state_values:
        if int(val) >= score_to_win:
            result = True
    return result


def is_loss(state):
    state_values = state.split(".")
    # if there is any open room on the board or any elements on the board are mergeable
    if "0" in state_values or is_mergeable(state_values):
        return False
    else:
        return True

# only implemented for 2x2 board


def is_mergeable(state_values):
    if (state_values[0] == state_values[1]
        or state_values[2] == state_values[3]
        or state_values[0] == state_values[2]
            or state_values[1] == state_values[3]):
        return True
    else:
        return False
