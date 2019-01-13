# @Author: voldikss
# @Date: 2019-01-13 10:11:31
# @Last Modified by: voldikss
# @Last Modified time: 2019-01-13 10:11:33

import numpy as np


def board_to_move(board, max_depth):
    '''入口函数'''
    results = []
    for direction in range(4):
        if np.any(board != merge(board, direction)):
            result = direction, deep_search(merge(board, direction), max_depth)
            results.append(result)
    # 已经不能再移动了的情况姑且返回方向为0
    if len(results) == 0:
        return 0
    return max(results, key=lambda x: x[1])[0]


def deep_search(board, depth, move=False):
    '''深度搜索'''

    # 到达最后一层搜索或游戏结束，则返回计算的值
    # 想起了短路运算...
    if depth == 0 or check_end(board):
        return eval_func(board)

    alpha = eval_func(board)

    if move:
        for direction in range(4):
            child = merge(board, direction)
            # 使用最大值而不用期望
            alpha = max(alpha, deep_search(child, depth - 1))
            # alpha += deep_search(child, depth - 1)
    else:
        alpha = 0
        zeros = np.where(board == 0)
        for i, j in zip(*zeros):
            child1 = board.copy()
            child2 = board.copy()
            child1[i, j] = 2
            child2[i, j] = 4
            alpha += (0.5 * deep_search(child1, depth - 1, True) +
                      0.5 * deep_search(child2, depth - 1, True))
        # 这里计算期望
        alpha /= len(zeros)
    return alpha


def eval_func(board):
    '''启发式算法，评估每一步的价值'''
    # 单调性 大的值尽量在外层
    mono_weight = np.array([[7, 6, 5, 4],
                            [6, 5, 4, 3],
                            [5, 4, 3, 2],
                            [4, 3, 2, 1]])
    monotonicity = np.sum(np.dot(mono_weight, board))

    # 平滑性 梯度要尽量小
    grad = np.gradient(board)
    smoothness = np.sum(np.abs(grad))
    return monotonicity - smoothness


def merge(board, direction):
    '''
    direction:
        0: left
        1: down
        2: right
        3: up
    '''

    def _merge(row):
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

    # treat all direction as left (by rotation)
    board = board.copy()
    board_to_left = np.rot90(board, -direction)
    for row in range(4):
        core = _merge(board_to_left[row])
        board_to_left[row, :len(core)] = core
        board_to_left[row, len(core):] = 0

    board = np.rot90(board_to_left, direction)
    return board


def check_end(board):
    '''
    检查棋盘状态
        0: continue
        1: lose/win
    '''
    where_empty = list(zip(*np.where(board == 0)))
    if where_empty:
        return 0
    else:
        return 1
