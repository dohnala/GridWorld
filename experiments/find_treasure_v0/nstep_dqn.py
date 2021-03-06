from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import OneHotEncoder
from env.tasks import FindTreasureTaskV0
from execution import SyncRunner
from experiments import Experiment
from networks import MLP
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV0(Experiment):
    """
    Experiment of N-step DQN agent for find_treasure_v0 task.
    """

    def __init__(self):
        super(FindTreasureV0, self).__init__()

    def define_task(self):
        return FindTreasureTaskV0()

    def define_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 1000),
                discount=0.95,
                n_step=8,
                target_sync=10))

    def define_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.93

    def train(self, env_fn, agent, seed):
        return SyncRunner(env_fn, agent, seed=seed).train(
            train_steps=5000,
            eval_every_steps=1000,
            eval_episodes=100,
            goal=self.define_goal)

    def eval(self, env_fn, agent, seed):
        return SyncRunner(env_fn, agent, seed=seed).eval(
            eval_episodes=100)


if __name__ == "__main__":
    FindTreasureV0().run()
