from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import LayerEncoder
from env.tasks import find_task
from execution import SyncRunner
from experiments import Experiment
from networks import CNN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV1(Experiment):
    """
    Experiment of N-step DQN agent for find_treasure_v1 task.
    """

    def __init__(self):
        super(FindTreasureV1, self).__init__()

    def define_task(self):
        return find_task("find_treasure_v1")

    def define_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=LayerEncoder(width, height, treasure_position=True),
                optimizer=AdamOptimizer(0.001),
                network=CNN(hidden_units=[128]),
                policy=EpsilonGreedyPolicy(1, 0.01, 4000),
                discount=0.95,
                n_step=16,
                target_sync=1000))

    def define_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def train(self, env, agent, seed):
        return SyncRunner(env, agent, seed=seed).train(
            train_episodes=10000,
            eval_episodes=100,
            eval_after=500,
            goal=self.define_goal)

    def eval(self, env, agent, seed):
        return SyncRunner(env, agent, seed=seed).eval(
            eval_episodes=100)


if __name__ == "__main__":
    FindTreasureV1().run()
