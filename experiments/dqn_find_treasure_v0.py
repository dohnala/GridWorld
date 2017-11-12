from agents import NStepDQNAgent
from encoders import OneHotEncoder
from experiments import Experiment
from networks import NN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV0(Experiment):
    """
    Experiment of DQN agent for find_treasure_v0 task.
    """

    def __init__(self):
        super(FindTreasureV0, self).__init__("find_treasure_v0")

    def create_agent(self, env):
        return NStepDQNAgent(env=env,
                             encoder=OneHotEncoder(env.width, env.height),
                             network=NN(),
                             optimizer=AdamOptimizer(0.01),
                             discount=0.95,
                             exploration_policy=EpsilonGreedyPolicy(0.5, 0.01, 150),
                             n_step=8)


if __name__ == "__main__":
    FindTreasureV0().run()
