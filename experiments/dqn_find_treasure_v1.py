from agents import NStepDQNAgent
from encoders import OneHotEncoder
from experiments import Experiment
from networks import NN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV1(Experiment):
    """
    Experiment of DQN agent for find_treasure_v0 task.
    """

    def __init__(self):
        super(FindTreasureV1, self).__init__("find_treasure_v1")

    def create_agent(self, env):
        return NStepDQNAgent(env=env,
                             encoder=OneHotEncoder(env.width, env.height, treasure_position=True),
                             network=NN(hidden_units=[128]),
                             optimizer=AdamOptimizer(0.002),
                             discount=0.95,
                             exploration_policy=EpsilonGreedyPolicy(1, 0.01, 2000),
                             n_step=8)


if __name__ == "__main__":
    FindTreasureV1().run()
