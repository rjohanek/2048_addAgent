# @Author: voldikss
# @Date: 2019-01-13 10:11:14
# @Last Modified by: voldikss
# @Last Modified time: 2019-01-13 15:53:08

import sys
sys.path.append(".")

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from game2048.agents import Agent
from game2048.game import Game
from game2048.expectimax import board_to_move
from keras.models import load_model
from util import CAND, MAP_TABLE, BATCH_SIZE, Guide, board_to_onehot
from model import build_model


class OmniAgent(Agent):
    """One agent to handle training."""

    def __init__(self, game, capacity):
        super().__init__(game)
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def train(self, begin_score, end_score, batch_size, num_epochs=5, n_tests=50, max_score_limit=True, eval_index=0.9):
        """如果有以保存的模型，则加载该模型进行训练"""
        if os.path.exists("model_%d_%d.h5" % (begin_score, end_score)):
            print("model exists, and will be loaded.")
            self.model = load_model("model_%d_%d.h5" % (begin_score, end_score))
        else:
            self.model = build_model(self.game.size, self.game.size, CAND)

        timer = 0
        while True:
            print("获取数据。。。")
            self.get_data(begin_score, end_score)
            X = []
            Y = []
            for guide in self.memory:
                X.append(guide.state)
                ohe_action = [0] * 4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            fitter = self.model.fit(
                np.array(X), np.array(Y),
                batch_size=batch_size,
                epochs=num_epochs,
                validation_split=0.1,
                shuffle=True,
                verbose=1)

            timer += 1

            # 每十次训练，保存精度，评估，模型等
            if not timer % 10 == 0:
                continue

            # 保存模型
            self.model.save("model_%d_%d.h5" % (begin_score, end_score))
            if timer % 100 == 0:
                self.model.save("model_%d_%d_100.h5" % (begin_score, end_score))

            # 保存评估分数
            if os.path.exists("eval_scores_%d_%d.json" % (begin_score, end_score)):
                with open("eval_scores_%d_%d.json" % (begin_score, end_score), 'r') as f:
                    eval_scores = json.load(f)
            else:
                eval_scores = []
            print("评测开始。评测次数：%d" % 50)
            start = time.perf_counter()
            eval_score = self.evaluate(begin_score, end_score, n_tests, max_score_limit, verbose=True)
            eval_scores.append(eval_score)
            print("评测完成。评测次数：%s  用时：%d s  平均得分：%.1f  目标分数：%d * %.1f = %.1f"
                  % (n_tests, time.perf_counter() - start, eval_score, end_score, eval_index, eval_index * end_score))
            with open("eval_scores_%d_%d.json" % (begin_score, end_score), 'w') as f:
                json.dump(eval_scores, f)

            # 画图
            plt.plot(range(len(eval_scores)), eval_scores)
            plt.title("eval_scores")
            plt.show()

            # 判断是否达标
            if eval_score > eval_index * end_score:
                self.model.save("model_%s_%s_success_%d.h5"
                                % (begin_score, end_score, eval_score))
                print("评测成功。已保存模型为 model_%s_%s_success_%d.h5"
                      % (begin_score, end_score, eval_score))
                quitq = input("评测已经达标，继续训练(y)或者退出(n)?")
                if quitq == 'n':
                    break
            else:
                print("评测失败。将继续训练。")

    def get_data(self, begin_score, end_score):
        '''获取训练数据'''
        # 清空memory
        self.memory = []
        while True:
            if len(self.memory) >= self.capacity:
                break
            self.reset_game(begin_score, end_score)
            while not self.game.end and begin_score <= self.game.score < end_score:
                direction = board_to_move(self.game.board)
                data = Guide(self.board_in_onehot, direction)
                self.memory.append(data)
                self.game.move(direction)
                if len(self.memory) % (self.capacity // 5) == 0:
                    print(len(self.memory), "/", self.capacity)

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
        #         print("Reset game from score %d to %d..." % (begin_score, end_score))
        new_board = np.zeros((self.game.size, self.game.size))
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
train_agent = OmniAgent(mygame, 2048)
train_agent.train((0, 1024), batch_size=BATCH_SIZE, num_epochs=5)
