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
        self.model.value_iteration()

    def step(self):
        state = convert_state(self.game.board)
        for i in range(len(self.model.states)):
            s = self.model.states[i]
            if s == state:
                return self.model.policies[i]
        # if state has never been seen, default behavior is left
        return 0


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

        # rewards only for terminal states +100 for win -100 for loss, 
        # and a calculation of score based on highest tile and future potential
        self.rewards = determine_rewards(self.states, game.score_to_win)

        # discounting value
        self.discount = discount

        # Set value iteration parameters
        self.k = 100  # Maximum number of iterations
        self.delta = 1e-400  # Error tolerance
        self.values = [0]*len(self.states)  # Initialize values
        self.policies = [None]*len(self.states)  # Initialize policy

    def value_iteration(self):
        # Repeat for k iterations or until convergence
        for num_iter in range(self.k):
            # Initialize update difference used to track convergence
            update_diff = 0

            # For each state in the state space
            for state_index in range(len(self.states)):
                # Get the state and set the maximum value to zero
                s = self.states[state_index]
                max_val = 0
                max_action = 1
                # Get the state value
                val = self.rewards[state_index]

                # For each possible action in the action space
                for action in self.actions:

                    # For each possible next state in the state space
                    '''If successors was working
                    for s_next in self.get_successors(self, self.states, self.transitions)
                    this would be a more efficient solution'''
                    for next_state_index in range(len(self.states)):
                        s_next = self.states[next_state_index]
                        # Check if there is a non-zero probability
                        if (s, s_next, action) in self.transitions:
                            # Add the product of the probability and the discounted value of the successor
                            val += self.transitions[(s, s_next, action)] * (
                                self.discount * self.values[next_state_index])

                    # Update the value if the new value is greater than the current value
                    # Update policy if the value is greater than the maximum value so far
                    if val > max_val:
                        max_val = val
                        max_action = action

                # Compute the difference between the updated and current value for the current state
                update_diff = max(update_diff, abs(
                    self.values[state_index] - max_val))
                # Update the value function with the updated values
                self.values[state_index] = max_val
                self.policies[state_index] = max_action

            # Check if convergence has been reached
            if update_diff < self.delta:
                break

    '''
    function iterates over all the available actions and computes the probability distribution over the next states for each action by querying the transition probabilities dictionary. 
    the get method of the dictionary is used to retrieve the probability of transitioning from state to next_state given action. 
    if the (state, action, next_state) key is not present in the dictionary, the probability is assumed to be 0.
    the function then generates a successor state for each action by randomly sampling from the probability distribution computed for that action using NumPy's np.random.choice method.
    the result is a list of successor states that can be reached from state by taking any of the available actions.

    '''
    # def get_successors(self, state, transition_probs):
    #     successors = []
    #     for action in range(self.actions):
    #         probs = [transition_probs.get((state, action, next_state), 0) for next_state in range(self.states)]
    #         next_state = np.random.choice(self.states, p=probs)
    #         successors.append(next_state)
    #     return successors

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
            rewards.append(100)
        elif is_loss(s):
            rewards.append(-100)
        else:
            rewards.append(calc_reward(s))
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

# try to distinguish boards based on their highest value, and their value potential in the future
def calc_reward(state):
    state_values = state.split(".")
    highest_score = max(int(i) for i in state_values)
    # if you have space on the board, that's good
    if "0" in state_values:
        highest_score +=1
    # if you can merge that's good
    if is_mergeable(state_values):
        highest_score +=1
    return highest_score