from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from keras.models import load_model
from task.util import board_to_onehot
import numpy as np
import re
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "task/model_0_1024.h5")
my_model = load_model(model_path)

direction_table = [Keys.ARROW_LEFT, Keys.ARROW_DOWN, Keys.ARROW_RIGHT, Keys.ARROW_UP]


def get_tiles():
    browser.get("http://sx349.github.io/2048easy/")
    tiles = browser.find_elements_by_class_name("tile")

    # In the case the page wasn't loaded entirely
    while len(tiles) == 0:
        browser.refresh()
        time.sleep(1)
        tiles = browser.find_elements_by_class_name("tile")

    board = np.zeros((4, 4))
    for tile in tiles:
        attrs = tile.get_attribute("class")
        value = int(re.findall(r"tile-(\d+?)\s", attrs)[0])
        pos = re.findall(r'tile-position-(\d+?)-(\d+?)', attrs)[0]
        xpos = int(pos[1]) - 1
        ypos = int(pos[0]) - 1
        board[xpos, ypos] = value

    return board


def predict(board, model):
    """
    :param board:
    :return: probability of every direction
    """
    if model == "cnn":
        board = np.array([board_to_onehot(board)])
        odds = my_model.predict(board)[0]
        directions = list(zip(*sorted(enumerate(odds), key=lambda x: x[1], reverse=True)))[0]
    else:
        from task.planning import predict
        odds = predict(board, max_depth=2)
        directions = list(zip(*sorted(enumerate(odds), key=lambda x: x[1])))[0]

    return directions


def mergable(board, direction):
    """Check whether the board can be merged"""

    def merge(row):
        '''merge the row, there may be some improvement'''
        non_zero = row[row != 0]  # remove zeros
        core = [None]
        for elem in non_zero:
            if core[-1] is None:
                core[-1] = elem
            elif core[-1] == elem:
                core[-1] = 2 * elem
                core.append(None)
            else:
                core.append(elem)
        if core[-1] is None:
            core.pop()
        return core

    # np.copy() is important
    board_to_left = np.rot90(board.copy(), -direction)
    for row in range(4):
        core = merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0

    # rotation to the original
    merged_board = np.rot90(board_to_left, direction)
    return np.any(merged_board != board)


def move(direction):
    body = browser.find_element_by_tag_name("body")
    body.send_keys(direction_table[direction])


if __name__ == '__main__':
    # Chrome driver should be installed first
    # See: http://chromedriver.chromium.org/downloads
    browser = webdriver.Chrome()
    browser.get("http://sx349.github.io/2048easy/")

    while True:
        board = get_tiles()

        # Test every direction, if mergable, merge and break
        # Default model is "cnn", also "planning" is available
        directions = predict(board, "cnn")
        for direction in directions:
            if mergable(board, direction):
                move(direction)
                break
        else:
            # If all the directions cannot be mergable, the game is over
            print("Game Is Over!")
            break

        # Sleepping for observation
        time.sleep(.5)
