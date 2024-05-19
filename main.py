from runner_train import Runner

from utils import load_json

config = load_json("./config/config.json")

runner = Runner()

runner.train()
runner.output()