from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import OneHotEncoder
from experiments import Experiment
from networks import NN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV1(Experiment):
    """
    Experiment of N-step DQN agent for find_treasure_v1 task.
    """

    def __init__(self):
        super(FindTreasureV1, self).__init__("find_treasure_v1")

    def create_agent(self, env):
        return NStepDQNAgent(
            env=env,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height, treasure_position=True),
                optimizer=AdamOptimizer(0.002),
                network=NN(hidden_units=[128]),
                policy=EpsilonGreedyPolicy(1, 0.01, 2000),
                discount=0.95,
                n_step=8))


if __name__ == "__main__":
    FindTreasureV1().run()
