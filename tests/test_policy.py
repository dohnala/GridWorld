import unittest

import numpy as np

from agent.policy import GreedyPolicy, EpsilonGreedyPolicy, BoltzmannPolicy


class GreedyPolicyTest(unittest.TestCase):
    def test_select_action(self):
        policy = GreedyPolicy()

        self.assertEqual(2, policy.select_action([1, 2, 3]))


class EpsilonGreedyPolicyTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_epsilon_after_create(self):
        policy = EpsilonGreedyPolicy(0, 0, 1)

        self.assertEqual(0, policy.epsilon)

    def test_select_action_for_low_epsilon(self):
        policy = EpsilonGreedyPolicy(0, 0, 1)

        self.assertEqual(2, policy.select_action([1, 2, 3]))

    def test_select_action_for_high_epsilon(self):
        policy = EpsilonGreedyPolicy(1, 1, 1)

        self.assertEqual(1, policy.select_action([1, 2, 3]))

    def test_select_action(self):
        policy = EpsilonGreedyPolicy(0.1, 0.1, 1)

        self.assertEqual(2, policy.select_action([1, 2, 3]))

    def test_epsilon_after_update(self):
        policy = EpsilonGreedyPolicy(1, 0.2, 2)

        self.assertEqual(1, policy.epsilon)

        policy.update()
        self.assertEqual(0.6, policy.epsilon)

        policy.update()
        self.assertEqual(0.2, round(policy.epsilon, 1))

        policy.update()
        self.assertEqual(0.2, round(policy.epsilon, 1))

    def test_epsilon_after_reset(self):
        policy = EpsilonGreedyPolicy(1, 0.2, 2)
        policy.update()

        self.assertEqual(0.6, policy.epsilon)

        policy.reset()

        self.assertEqual(1, policy.epsilon)


class BoltzmannPolicyTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_temperature_after_create(self):
        policy = BoltzmannPolicy(1, 1, 1)

        self.assertEqual(1, policy.temperature)

    def test_select_action_for_high_temperature(self):
        policy = BoltzmannPolicy(100, 100, 1)

        self.assertEqual(1, policy.select_action([1, 2, 3]))

    def test_select_action_for_low_temperature(self):
        policy = BoltzmannPolicy(1, 1, 1)

        self.assertEqual(2, policy.select_action([1, 2, 3]))

    def test_select_action(self):
        policy = BoltzmannPolicy(10, 10, 1)

        self.assertEqual(1, policy.select_action([1, 2, 3]))

    def test_temperature_after_update(self):
        policy = BoltzmannPolicy(10, 2, 2)

        self.assertEqual(10, policy.temperature)

        policy.update()
        self.assertEqual(6, policy.temperature)

        policy.update()
        self.assertEqual(2, policy.temperature)

        policy.update()
        self.assertEqual(2, policy.temperature)

    def test_temperature_after_reset(self):
        policy = BoltzmannPolicy(10, 2, 2)
        policy.update()

        self.assertEqual(6, policy.temperature)

        policy.reset()

        self.assertEqual(10, policy.temperature)
