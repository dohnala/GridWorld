from agents import AsyncNStepDQNAgent, AsyncNStepDQNAgentConfig as Config
from encoders import LayerEncoder
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
        super(FindTreasureV1, self).__init__("find_treasure_v1")

    def create_agent(self, width, height, num_actions):
        return AsyncNStepDQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=LayerEncoder(width, height, treasure_position=True),
                optimizer=SharedAdamOptimizer(0.001),
                network=CNN(hidden_units=[128]),
                policy=EpsilonGreedyPolicy(1, 0.01, 2000),
                discount=0.95,
                n_step=16))

    def create_runner(self, env_creator, agent_creator):
        return AsyncRunner(env_creator, agent_creator, num_workers=4)

    def termination_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90


if __name__ == "__main__":
    FindTreasureV1().run()
