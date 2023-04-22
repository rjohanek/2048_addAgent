# EE369-2048-AI

For SJTU [EE369](https://github.com/duducheng/2048-api) final project.

Use supervised learning (imitation learning) and tree searching approaches to solve the game of 2048.

## Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`task/`](task/): the implementation of supervised learning and tree searching.
    * [`agents.py`](task/agents.py): the `Agent` classes of supervised learning and tree searching.
    * [`model.py`](task/model.py): the convolutional neural network model.
    * [`offline_training.py`](task/offline_training.py): offline method for training.
    * [`online_training.py`](task/online_training.py): online method for training.
    * [`planning.py`](task/planning.py): the tree searching approach solution of the game.
    * [`util.py`](task/util.py): tools to process the game board.
    * [`model_0_1024.h5`](task/model_0_1024.h5): the dumped CNN model.
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate the self-defined agent.


## To run the evaluation of the agents

* To evaluate the supervised learning model, run
```bash
# Will play the game for 50 times and return the average score
python evaluate.py --agent=cnnagent
```
P.S. Currently the max score is 1024, the average score is 541.44.

* To evaluate the tree searching method, run
```bash
python evaluate.py --agent=pagent
```
P.S. With the depth set to 3, the planning method can reach the score 2048.


## To run the web app
```
python webapp.py 
```
You can also specify an agent by adding `--agent`. `cnnagent`, `pagent`, `emagent` are usable, `RandomAgent` by default.

For example, run the web app with the planning agent
```bash
python webapp.py --agent=pagen
```

## To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```

![demo](preview2048.gif)

## LICENSE
The code is under Apache-2.0 License.
pp.