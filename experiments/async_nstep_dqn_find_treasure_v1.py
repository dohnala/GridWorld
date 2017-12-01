from agents import AsyncNStepDQNAgent, AsyncNStepDQNAgentConfig as Config
from encoders import LayerEncoder
from env.tasks import find_task
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
        return find_task("find_treasure_v1")

    def define_agent(self, width, height, num_actions):
        return AsyncNStepDQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=LayerEncoder(width, height, treasure_position=True),
                optimizer=SharedAdamOptimizer(0.001),
                network=CNN(hidden_units=[128]),
                policy=EpsilonGreedyPolicy(1, 0.01, 1000),
                discount=0.95,
                n_step=16))

    def define_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def train(self, env, agent):
        return AsyncRunner(env, agent, 4).train(
            train_episodes=10000,
            eval_episodes=100,
            eval_after_sec=1,
            goal=self.define_goal)

    def eval(self, env, agent):
        return AsyncRunner(env, agent, 4).eval(
            eval_episodes=100)


if __name__ == "__main__":
    FindTreasureV1().run()
