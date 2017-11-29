from agents import DQNAgent, DQNAgentConfig as Config
from encoders import OneHotEncoder
from env.tasks import FindTreasureTask, find_task
from networks import NN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy
from tests.agent.test_agent import AgentTestCases


class DQNAgentWithoutMemoryTest(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def create_agent(self, width, height, num_actions):
        return DQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                capacity=1,
                batch_size=1))


class DQNAgentWithMemoryTest(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def create_agent(self, width, height, num_actions):
        return DQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                capacity=100,
                batch_size=16))


class DQNAgentWithTargetSyncTest(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def create_agent(self, width, height, num_actions):
        return DQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                capacity=1,
                batch_size=1,
                target_sync=10))


class DQNAgentForFindTreasureV0Test(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_task(self):
        return find_task("find_treasure_v0")

    def create_agent(self, width, height, num_actions):
        return DQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                capacity=1000,
                batch_size=32,
                target_sync=10))