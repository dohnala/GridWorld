from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import OneHotEncoder
from execution import Runner
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

    def create_runner(self, env):
        agent = NStepDQNAgent(
            num_actions=env.num_actions,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                n_step=8))

        return Runner(env, agent)

    def termination_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.93


if __name__ == "__main__":
    FindTreasureV0().run()
