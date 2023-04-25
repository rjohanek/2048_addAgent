from game2048.game import Game
from game2048.displays import Display
import matplotlib.pyplot as plt
import sys
from collections import Counter

# copied from evaluate.py
# added calculate_probabilities
# and changed single_run to call play_learn
# changed main to call learning agent by default and keep track of counts


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    counts = agent.play_learn(verbose=True)
    return counts


def calculate_probabilities(counts):
    # sum up the counts for all state pairs that originate at the same state to be denominator for each state
    occurrence_of_each_state = Counter()
    for entry in counts.elements():
        state1 = entry[0]
        value = counts[entry]
        occurrence_of_each_state.update([state1]*value)

    # divide each state pair count by its corresponding denominator
    transition_probabilities = {}
    for entry in counts.elements():
        state1 = entry[0]
        denom = occurrence_of_each_state[state1]
        if denom > 0:
            transition_probabilities[entry] = counts[entry] / denom
        else:
            transition_probabilities[entry] = 0

    # return probabilities that represent the liklihood of going to the second state in the pair, given the first
    # and return a list of all states explored
    return {"states": occurrence_of_each_state.keys(), "probs": transition_probabilities}


if __name__ == '__main__':
    GAME_SIZE = 2
    SCORE_TO_WIN = 32
    N_TESTS = 50

    if len(sys.argv) == 2:
        agent_name = sys.argv[1].split("=")[-1]
        if agent_name == "learning":
            from task.agents import LearningAgent as TestAgent
        else:
            print("WARNING: Agent class doesn't exist.")
    else:
        # default
        from task.agents import LearningAgent as TestAgent

    total_counts = Counter()
    for i in range(N_TESTS):
        print("N_TESTS for :%d" % i)
        counts = single_run(GAME_SIZE, SCORE_TO_WIN,
                            AgentClass=TestAgent)
        total_counts.update(counts)

    with open("learned_states_probs.txt", "a") as f:
        print(calculate_probabilities(total_counts), file=f)
