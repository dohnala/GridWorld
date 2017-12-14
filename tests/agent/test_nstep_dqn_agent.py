from agents import NStepDQNAgent, NStepDQNAgentConfig as Config
from encoders import OneHotEncoder
from env.tasks import FindTreasureTask, FindTreasureTaskV0
from execution import SyncRunner
from networks import MLP
from optimizers import AdamOptimizer
from policies import EpsilonGreedyPolicy
from tests.agent import AgentTestCases


class SimpleOneStepDQNAgentTest(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def define_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 1000),
                discount=0.95,
                n_step=1))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            train_steps=5000,
            eval_every_steps=1000,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)


class SimpleNStepDQNAgentTest(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def define_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 1000),
                discount=0.95,
                n_step=8))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            train_steps=5000,
            eval_every_steps=1000,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)


class SimpleDQNAgentWithTargetSyncTest(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTask(width=4, height=4, episode_length=20, treasure_position=(2, 3))

    def define_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 1000),
                discount=0.95,
                n_step=1,
                target_sync=500))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.90

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            train_steps=5000,
            eval_every_steps=1000,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)


class NStepDQNAgentForFindTreasureV0Test(AgentTestCases.AgentTestCase):
    def define_task(self):
        return FindTreasureTaskV0()

    def define_agent(self, width, height, num_actions):
        return NStepDQNAgent(
            config=Config(
                num_actions=num_actions,
                encoder=OneHotEncoder(width, height),
                optimizer=AdamOptimizer(0.01),
                network=MLP(),
                policy=EpsilonGreedyPolicy(1, 0.01, 1500),
                discount=0.95,
                n_step=8,
                target_sync=10))

    def define_train_goal(self, result):
        return result.get_accuracy() == 100 and result.get_mean_reward() >= 0.93

    def define_eval_goal(self, result):
        return result.accuracy == 100 and result.reward >= 0.90

    def train(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).train(
            train_steps=5000,
            eval_every_steps=1000,
            eval_episodes=100,
            goal=self.define_train_goal)

    def eval(self, env_fn, agent):
        return SyncRunner(env_fn, agent, self.seed).eval(
            eval_episodes=100)
