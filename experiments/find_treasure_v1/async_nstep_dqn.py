from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import LayerEncoder
from env.tasks import FindTreasureTaskV1
from execution import AsyncRunner
from experiments import Experiment
from networks import CNN
from optimizers import SharedAdamOptimizer
from policies import EpsilonGreedyPolicy


class FindTreasureV1(Experiment):
    """
    Experiment of Asynchronous N-step DQN agent for find_treasure_v1 task.
    """

    def __init__(self):
        super(FindTreasureV1, self).__init__()

    def define_task(self):
        return FindTreasureTaskV1()

    def define_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=LayerEncoder(width, height, treasure_position=True),
                optimizer=SharedAdamOptimizer(0.001),
                network=CNN(hidden_units=[128]),
                policy=EpsilonGreedyPolicy(1, 0.01, 25000),
                discount=0.95,
                n_step=16))

    def define_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def train(self, env_fn, agent, seed):
        return AsyncRunner(env_fn, agent, num_processes=4, seed=seed).train(
            train_steps=400000,
            eval_every_sec=1,
            eval_episodes=100,
            goal=self.define_goal)

    def eval(self, env_fn, agent, seed):
        return AsyncRunner(env_fn, agent, num_processes=4, seed=seed).eval(
            eval_episodes=100)


if __name__ == "__main__":
    FindTreasureV1().run()
