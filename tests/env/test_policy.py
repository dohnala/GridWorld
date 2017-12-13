import unittest

import numpy as np

from policies import GreedyPolicy, EpsilonGreedyPolicy


class GreedyPolicyTest(unittest.TestCase):
    def test_select_action(self):
        policy = GreedyPolicy()

        self.assertEqual(2, policy.select_action([1, 2, 3], 0))


class EpsilonGreedyPolicyTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_select_action_for_low_epsilon(self):
        policy = EpsilonGreedyPolicy(0, 0, 1)

        self.assertEqual(2, policy.select_action([1, 2, 3], 0))

    def test_select_action_for_high_epsilon(self):
        policy = EpsilonGreedyPolicy(1, 1, 1)

        self.assertEqual(1, policy.select_action([1, 2, 3], 0))

    def test_select_action(self):
        policy = EpsilonGreedyPolicy(0.1, 0.1, 1)

        self.assertEqual(2, policy.select_action([1, 2, 3], 0))

    def test_select_action_steps(self):
        policy = EpsilonGreedyPolicy(1, 0, 2)

        self.assertEqual(1, policy.select_action([1, 2, 3], 0))
        self.assertEqual(2, policy.select_action([1, 2, 3], 1))
        self.assertEqual(2, policy.select_action([1, 2, 3], 2))
        self.assertEqual(2, policy.select_action([1, 2, 3], 3))
