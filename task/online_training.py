# @Author: voldikss
# @Date: 2019-01-13 10:11:23
# @Last Modified by: voldikss
# @Last Modified time: 2019-01-13 15:54:03

import sys
sys.path.append(".")

import os
import random
import numpy as np
from game2048.agents import Agent
from game2048.game import Game
from game2048.expectimax import board_to_move
from keras.models import load_model
from model import build_model
from util import CAND, MAP_TABLE, BATCH_SIZE, Guide, board_to_onehot


class OmniAgent(Agent):
    """One agent to handle training."""

    def __init__(self, game, capacity):
        super().__init__(game)
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def train(self, begin_score, end_score):
        """如果有以保存的模型，则加载该模型进行训练"""
        if os.path.exists("model_%d_%d.h5" % (begin_score, end_score)):
            print("model exists, and will be loaded.")
            self.model = load_model("model_%d_%d.h5" % (begin_score, end_score))
        else:
            self.model = build_model(self.game.size, self.game.size, CAND)

        timer = 0
        while True:
            # 先重置游戏到所要训练区间
            self.reset_game(begin_score, end_score)
            while not self.game.end:
                self.move()
            print("分数：", self.game.score, end='\t')

            if len(self.memory) < BATCH_SIZE:
                # 不满足一次训练所需数量，继续获取数据
                continue
            guides = random.sample(self.memory, BATCH_SIZE)
            X = []
            Y = []
            for guide in guides:
                X.append(guide.state)
                ohe_action = [0] * 4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            loss, acc = self.model.train_on_batch(np.array(X), np.array(Y))
            print('第 %d 轮 \t loss:%.3f \t acc:%.3f' % (timer, float(loss), float(acc)))

            timer += 1

            if timer % 10 == 0:
                self.model.save("model_%d_%d.h5" % (begin_score, end_score))

            if timer % 100 == 0:
                self.model.save("model_%d_%d_100.h5" % (begin_score, end_score))

    def move(self):
        '''由我的模型移动方块，但根据 ExpectiMax 的决策保存训练数据'''
        self.push(self.board_in_onehot, board_to_move(self.game.board))
        self.game.move(self.predict())

    def push(self, *args):
        '''向 memory 里面放数据'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = Guide(*args)
        self.pos = (self.pos + 1) % self.capacity

    def reset_game(self, begin_score, end_score):
        '''
        重置游戏
        e.g.
        [0,256],[0,1024]...
        [256,512]
        [512,1024]
        '''
        if not begin_score in MAP_TABLE:
            raise AssertionError("init_max_score must be a number in %s"
                                 % list(MAP_TABLE.keys()))
        # 从0开始
        new_board = np.zeros((self.game.size, self.game.size))
        # 不是从0开始
        if begin_score > 2:
            other_scores = [i for i in MAP_TABLE if i < begin_score]
            other_scores = np.random.choice(other_scores, int(len(other_scores) * 0.8), replace=True)
            locations = np.random.choice(16, 1 + len(other_scores), replace=False)
            new_board[locations // 4, locations % 4] = np.append(other_scores, begin_score)

        self.game.board = new_board
        self.game._maybe_new_entry()
        self.game._maybe_new_entry()
        self.game.__end = 0
        self.game.score_to_win = end_score

    @property
    def board_in_onehot(self):
        '''one-hot编码形式的棋盘数据'''
        return board_to_onehot(self.game.board)

    def predict(self):
        '''由模型对给定棋盘做出决策'''
        board = np.array([self.board_in_onehot])
        direction = int(self.model.predict(board).argmax())
        return direction

    def evaluate(self, begin_score, end_score, n_tests, max_score_limit=True, verbose=False):
        '''在训练过程中对当前模型进行评测'''
        scores = []
        for i in range(n_tests):
            if not max_score_limit:
                self.reset_game(begin_score, np.inf)
            else:
                self.reset_game(begin_score, end_score)
            while not self.game.end:
                direction = self.predict()
                self.game.move(direction)
            scores.append(self.game.score)
        if verbose:
            print(scores)
        score = sum(scores) / len(scores)
        return score


mygame = Game(enable_rewrite_board=True)
train_agent = OmniAgent(mygame, 32768)
train_agent.train(0, 1024)
