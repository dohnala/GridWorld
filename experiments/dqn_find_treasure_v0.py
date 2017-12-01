from agents import DQNAgent, DQNAgentConfig as Config
from encoders import OneHotEncoder
from env.tasks import find_task
from execution import SyncRunner
from experiments import Experiment
from networks import NN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV0(Experiment):
    """
    Experiment of DQN agent for find_treasure_v0 task.
    """

    def __init__(self):
        super(FindTreasureV0, self).__init__()

    def define_task(self):
        return find_task("find_treasure_v0")

    def define_agent(self, width, height, num_actions):
        return DQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.001),
                network=NN(),
                policy=EpsilonGreedyPolicy(1, 0.01, 100),
                discount=0.95,
                capacity=1000,
                batch_size=32,
                target_sync=10))

    def define_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def train(self, env, agent):
        return SyncRunner(env, agent).train(
            train_episodes=2000,
            eval_episodes=100,
            eval_after=200,
            goal=self.define_goal)

    def eval(self, env, agent):
        return SyncRunner(env, agent).eval(
            eval_episodes=100)


if __name__ == "__main__":
    FindTreasureV0().run()
