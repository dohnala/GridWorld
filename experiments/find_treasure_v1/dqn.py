from agents import DQNAgent, DQNAgentConfig as Config
from encoders import LayerEncoder
from env.tasks import FindTreasureTaskV1
from execution import SyncRunner
from experiments import Experiment
from networks import CNN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV1(Experiment):
    """
    Experiment of DQN agent for find_treasure_v1 task.
    """

    def __init__(self):
        super(FindTreasureV1, self).__init__()

    def define_task(self):
        return FindTreasureTaskV1()

    def define_agent(self, width, height, num_actions):
        return DQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=LayerEncoder(width, height, treasure_position=True),
                optimizer=AdamOptimizer(0.001),
                network=CNN(hidden_units=[128]),
                policy=EpsilonGreedyPolicy(1, 0.01, 50000),
                discount=0.95,
                capacity=10000,
                batch_size=8,
                target_sync=100,
                double_q=True))

    def define_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def train(self, env, agent, seed):
        return SyncRunner(env, agent, seed=seed).train(
            max_steps=100000,
            eval_every_steps=1000,
            eval_episodes=100,
            goal=self.define_goal)

    def eval(self, env, agent, seed):
        return SyncRunner(env, agent, seed=seed).eval(
            eval_episodes=100)


if __name__ == "__main__":
    FindTreasureV1().run()
