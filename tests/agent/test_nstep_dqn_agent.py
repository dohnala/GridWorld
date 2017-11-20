from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import OneHotEncoder
from env.env import GridWorldEnv
from env.tasks.find_treasure import FindTreasureTask
from networks import NN
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy
from tests.agent.test_agent import AgentTestCases


class SimpleOneStepDQNAgentTest(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def create_env(self):
        return GridWorldEnv(FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3)))

    def create_agent(self, env):
        return NStepDQNAgent(
            num_actions=env.num_actions,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height),
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

    def create_env(self):
        return GridWorldEnv(FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3)))

    def create_agent(self, env):
        return NStepDQNAgent(
            num_actions=env.num_actions,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                n_step=8))


class SimpleDQNAgentWithTargetSyncTest(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def eval_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.85

    def create_env(self):
        return GridWorldEnv(FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3)))

    def create_agent(self, env):
        return NStepDQNAgent(
            num_actions=env.num_actions,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height),
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

    def create_env(self):
        return GridWorldEnv.for_task_name("find_treasure_v0")

    def create_agent(self, env):
        return NStepDQNAgent(
            num_actions=env.num_actions,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height),
                optimizer=AdamOptimizer(0.01),
                network=NN(),
                policy=EpsilonGreedyPolicy(0.5, 0.01, 500),
                discount=0.95,
                n_step=8,
                target_sync=10))


class NStepDQNAgentForFindTreasureV1Test(AgentTestCases.AgentTestCase):
    def train_cond(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.98

    def eval_cond(self, result):
        return result.get_accuracy() >= 95 and result.get_mean_reward() >= 0.95

    def create_env(self):
        return GridWorldEnv.for_task_name("find_treasure_v1")

    def create_agent(self, env):
        return NStepDQNAgent(
            num_actions=env.num_actions,
            config=Config(
                encoder=OneHotEncoder(env.width, env.height, treasure_position=True),
                optimizer=AdamOptimizer(0.002),
                network=NN(hidden_units=[128]),
                policy=EpsilonGreedyPolicy(1, 0.01, 2000),
                discount=0.95,
                n_step=8,
                target_sync=100))
