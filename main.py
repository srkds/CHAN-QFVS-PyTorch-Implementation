from runner_train import Runner

from utils import load_json

config = load_json("./config/config.json")

runner = Runner(config, [1,2,3], 4) # train video list, test video

runner.train()
# runner.output()