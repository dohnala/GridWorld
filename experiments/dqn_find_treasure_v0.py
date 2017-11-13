from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import OneHotEncoder
from experiments import Experiment
from networks import NN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV0(Experiment):
    """
    Experiment of N-step DQN agent for find_treasure_v0 task.
    """

    def __init__(self):
        super(FindTreasureV0, self).__init__("find_treasure_v0")

    def create_agent(self, env):
        return NStepDQNAgent(
            env=env,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 150),
                discount=0.95,
                n_step=8))


if __name__ == "__main__":
    FindTreasureV0().run()
