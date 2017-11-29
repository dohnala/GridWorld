import os
import unittest

from env.env import GridWorldEnv
from execution import SyncRunner


class AgentTestCases:
    class AgentTestCase(unittest.TestCase):
        """
        Base test case for testing agents.
        """

        def tearDown(self):
            os.remove("agent.ckp")

        def train_cond(self, result):
            """
            Termination condition for training.

            :param result: training result
            :return: True if training should be terminated
            """
            return False

        def eval_cond(self, result):
            """
            Termination condition for evaluation.

            :param result: evaluation result
            :return: True if evaluation is sufficient
            """
            return False

        def create_task(self):
            """
            Create task.

            :return: task
            """
            pass

        def create_agent(self, width, height, num_actions):
            """
            Create agent.

            :param width: width
            :param height: height
            :param num_actions: num_actions
            :return: agent
            """
            pass

        def test_agent(self):
            """
            Train agent on given environment until training termination condition is passed or
            number of training episodes is reached. After that, check if agent passes evaluation
            condition, save its state, and check that restored agent passes evaluation condition, too.

            :return: None
            """
            task = self.create_task()

            def env_creator():
                return GridWorldEnv(task, seed=1)

            def agent_creator():
                return self.create_agent(task.width, task.height, len(task.get_actions()))

            def save(run, agent):
                agent.save("agent.ckp")

            runner = SyncRunner(env_creator, agent_creator, seed=1)

            result = runner.train(
                train_episodes=3000,
                eval_episodes=100,
                eval_after=200,
                termination_cond=self.train_cond,
                after_run=save)

            self.assertTrue(self.eval_cond(result), "accuracy:{:7.2f}%, reward:{:6.2f}".format(
                result.get_accuracy(), result.get_mean_reward()))

            def load_agent_creator():
                agent = self.create_agent(task.width, task.height, len(task.get_actions()))
                agent.load("agent.ckp")

                return agent

            runner = SyncRunner(env_creator, load_agent_creator, seed=1)

            result = runner.eval(eval_episodes=100)

            self.assertTrue(self.eval_cond(result), "accuracy:{:7.2f}%, reward:{:6.2f}".format(
                result.get_accuracy(), result.get_mean_reward()))
