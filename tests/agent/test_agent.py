import unittest

import os

from execution import Runner


class AgentTestCases:
    class AgentTestCase(unittest.TestCase):
        """
        Base test case for testing agents.
        """

        def tearDown(self):
            os.remove("model.ckp")

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

        def create_env(self):
            """
            Create environment.

            :return: environment
            """
            pass

        def create_agent(self, env):
            """
            Create agent.

            :param env: environment
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
            env = self.create_env()

            runner = Runner(env, self.create_agent)

            result = runner.run(
                train_episodes=3000,
                eval_episodes=100,
                eval_after=500,
                log_after=100,
                termination_cond=self.train_cond)

            runner.agent.save("model.ckp")

            self.assertTrue(self.eval_cond(result), "accuracy:{:7.2f}%, reward:{:6.2f}".format(
                result.get_accuracy(), result.get_mean_reward()))

            def agent_creator(_env):
                agent = self.create_agent(_env)
                agent.load("model.ckp")

                return agent

            runner = Runner(env, agent_creator)

            result = runner.run(
                train_episodes=0,
                eval_episodes=100,
                eval_after=0,
                log_after=100,
                termination_cond=self.eval_cond)

            self.assertTrue(self.eval_cond(result), "accuracy:{:7.2f}%, reward:{:6.2f}".format(
                result.get_accuracy(), result.get_mean_reward()))
