from agents import DQNAgent, DQNAgentConfig as Config
from encoders import OneHotEncoder
from env.tasks import FindTreasureTask, FindTreasureTaskV0
from execution import SyncRunner
from networks import MLP
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy
from tests.agent import AgentTestCases


class DQNAgentWithoutMemoryTest(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def define_agent(self, width, height, num_actions):
        return DQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 500),
                discount=0.95,
                capacity=1,
                batch_size=1))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            max_steps=1000,
            eval_every_steps=100,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)


class DQNAgentWithMemoryTest(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def define_agent(self, width, height, num_actions):
        return DQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 500),
                discount=0.95,
                capacity=100,
                batch_size=16))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            max_steps=1000,
            eval_every_steps=100,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)


class DQNAgentWithTargetSyncTest(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def define_agent(self, width, height, num_actions):
        return DQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 500),
                discount=0.95,
                capacity=1,
                batch_size=1,
                target_sync=10))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            max_steps=1000,
            eval_every_steps=100,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)


class DQNAgentWithDoubleQ(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def define_agent(self, width, height, num_actions):
        return DQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 500),
                discount=0.95,
                capacity=1,
                batch_size=1,
                target_sync=10,
                double_q=True))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            max_steps=1000,
            eval_every_steps=100,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)


class DQNAgentForFindTreasureV0Test(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTaskV0()

    def define_agent(self, width, height, num_actions):
        return DQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.001),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 2000),
                discount=0.95,
                capacity=1000,
                batch_size=32,
                target_sync=10))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            max_steps=5000,
            eval_every_steps=1000,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)
