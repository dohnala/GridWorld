import os
import random
import unittest

import numpy as np
import torch

from env.env import GridWorldEnv


class AgentTestCases:
    class AgentTestCase(unittest.TestCase):
        """
        Base test case for testing agents.
        """

        def __init__(self, methodName='runTest', seed=1):
            super(AgentTestCases.AgentTestCase, self).__init__(methodName)

            self.seed = seed

        def setUp(self):
            self.set_seed(self.seed)

        def tearDown(self):
            os.remove("agent.ckp")

        def define_task(self):
            """
            Create task.

            :return: task
            """
            pass

        def define_agent(self, width, height, num_actions):
            """
            Create agent.

            :param width: width
            :param height: height
            :param num_actions: num_actions
            :return: agent
            """
            pass

        def define_train_goal(self, result):
            """
            Define goal for training.

            :param result: training result
            :return: True if training should be terminated
            """
            return False

        def define_eval_goal(self, result):
            """
            Define goal for evaluation.

            :param result: evaluation result
            :return: True if evaluation is sufficient
            """
            return False

        def train(self, env, agent):
            """
            Train agent on environment.

            :param env: environment
            :param agent: agent
            :return: result
            """
            pass

        def eval(self, env, agent):
            """
            Evaluate agent on environment.

            :param env: environment
            :param agent: agent
            :return: result
            """
            pass

        def test_agent(self):
            """
            Train agent on given environment until training goal is reached or
            number of training episodes is exceeded. After that, check if agent passes evaluation
            goal, save its state, and check that restored agent passes goal condition, too.

            :return: None
            """
            # Create task
            task = self.define_task()

            # Create environment
            env = GridWorldEnv(task, seed=self.seed)

            # Create train agent
            train_agent = self.define_agent(task.width, task.height, len(task.get_actions()))

            # Train agent
            train_result = self.train(env, train_agent)

            # Save agent
            train_agent.save("agent.ckp")

            # Evaluate agent
            eval_result = self.eval(env, train_agent)

            # Assert that agent passes evaluation goal
            self.assertTrue(self.define_eval_goal(eval_result), "accuracy:{:7.2f}%, reward:{:6.2f}".format(
                train_result.accuracy, train_result.reward))

            # Create evaluation agent
            eval_agent = self.define_agent(task.width, task.height, len(task.get_actions()))

            # Restore evaluation agent
            eval_agent.load("agent.ckp")

            # Evaluate agent
            eval_result = self.eval(env, eval_agent)

            # Assert that evaluation agent passes evaluation goal
            self.assertTrue(self.define_eval_goal(eval_result), "accuracy:{:7.2f}%, reward:{:6.2f}".format(
                eval_result.accuracy, eval_result.reward))

        @staticmethod
        def set_seed(seed):
            """
            Set random seed.

            :param seed: seed
            :return: None
            """
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)