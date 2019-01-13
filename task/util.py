# @Author: voldikss
# @Date: 2019-01-13 10:11:38
# @Last Modified by: voldikss
# @Last Modified time: 2019-01-13 23:31:00

import numpy as np
from collections import namedtuple

# 16 is better...
CAND = 12
MAP_TABLE = {2 ** i: i for i in range(1, CAND)}
MAP_TABLE[0] = 0
BATCH_SIZE = 1024

Guide = namedtuple("Guides", ("state", "action"))


def board_to_onehot(board):
    '''将棋盘数据转化为one-hot编码'''
    ret = np.zeros(shape=(4, 4, CAND), dtype=float)
    for r in range(4):
        for c in range(4):
            ret[r, c, MAP_TABLE[board[r, c]]] = 1
    return ret
