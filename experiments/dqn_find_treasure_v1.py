from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import OneHotEncoder
from execution import Runner
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

    def create_runner(self, env):
        agent = NStepDQNAgent(
            num_actions=env.num_actions,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height, treasure_position=True),
                optimizer=AdamOptimizer(0.002),
                network=NN(hidden_units=[128]),
                policy=EpsilonGreedyPolicy(1, 0.01, 2000),
                discount=0.95,
                n_step=8,
                target_sync=100))

        return Runner(env, agent)

    def termination_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.98

if __name__ == "__main__":
    FindTreasureV1().run()
