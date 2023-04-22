from flask import Flask, jsonify, request
import sys


def get_flask_app(game, agent):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return app.send_static_file('board.html')

    @app.route("/board", methods=['GET', 'POST'])
    def get_board():
        direction = -1
        control = "USER"
        if request.method == "POST":
            direction = request.json
            if direction == -1:
                direction = agent.step()
                control = 'AGENT'
            game.move(direction)
        return jsonify({"board": game.board.tolist(),
                        "score": game.score,
                        "end": game.end,
                        "direction": direction,
                        "control": control})

    return app


if __name__ == "__main__":
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    APP_PORT = 5005
    APP_HOST = "localhost"

    if len(sys.argv) == 2:
        agent_name = sys.argv[1].split("=")[-1]
        if agent_name == "emagent":
            from game2048.agents import ExpectiMaxAgent as TestAgent
        elif agent_name == "pagent":
            from task.agents import PlanningAgent as TestAgent
        elif agent_name == "cnnagent":
            from task.agents import CNNAgent as TestAgent
        elif agent_name == "markov_value":
            GAME_SIZE = 2
            SCORE_TO_WIN = 32
            from task.agents import MarkovAgent as TestAgent
        else:
            print("WARNING: Agent class doesn't exist.")
    else:
        from game2048.agents import RandomAgent as TestAgent
        print("WARNING: You are now using a RandomAgent.")

    from game2048.game import Game
    game = Game(size=GAME_SIZE, score_to_win=SCORE_TO_WIN)

    agent = TestAgent(game=game)

    print("Run the webapp at http://<any address for your local host>:%s/" % APP_PORT)

    app = get_flask_app(game, agent)
    app.run(port=APP_PORT, threaded=False, host=APP_HOST)  # IMPORTANT: `threaded=False` to ensure correct behavior
