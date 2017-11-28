from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import OneHotEncoder
from env.tasks import find_task, FindTreasureTask
from networks import NN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy
from tests.agent.test_agent import AgentTestCases


class SimpleOneStepDQNAgentTest(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def create_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                n_step=1))


class SimpleNStepDQNAgentTest(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def create_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                n_step=8))


class SimpleDQNAgentWithTargetSyncTest(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def create_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                n_step=1,
                target_sync=10))


class NStepDQNAgentForFindTreasureV0Test(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.95

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_task(self):
        return find_task("find_treasure_v0")

    def create_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            num_actions=num_actions,
            config=Config(
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                n_step=8,
                target_sync=10))
