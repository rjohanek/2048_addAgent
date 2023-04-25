from game2048.game import Game
from game2048.displays import Display
import matplotlib.pyplot as plt
import sys


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50

    if len(sys.argv) == 2:
        agent_name = sys.argv[1].split("=")[-1]
        if agent_name == "emagent":
            from game2048.agents import ExpectiMaxAgent as TestAgent
        elif agent_name == "pagent":
            from task.agents import PlanningAgent as TestAgent
        elif agent_name == "cnnagent":
            from task.agents import CNNAgent as TestAgent
        if agent_name == "markov_value":
            GAME_SIZE = 2
            SCORE_TO_WIN = 32
            from task.agents import MarkovAgent as TestAgent
        else:
            print("WARNING: Agent class doesn't exist.")
    else:
        # default
        from task.agents import CNNAgent as TestAgent

    '''====================
    Use ExpectiMaxAgent here.'''
    # from game2048.agents import ExpectiMaxAgent as TestAgent
    '''====================
    Use PlanningAgent here.'''
    # from task.agents import PlanningAgent as TestAgent
    '''====================
    Use CNNAgent here.'''
    # from task.agents import CNNAgent as TestAgent
    '''===================='''

    scores = []
    for i in range(N_TESTS):
        print("N_TESTS for :%d" % i)
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)

    # plt.plot(scores)
    # plt.xlabel("Loops")
    # plt.ylabel("Score")
    # plt.yticks([2**i for i in range(5,11):])
    # plt.title("Score Distribution Over %d Tests" % N_TESTS)
    # plt.show()

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
