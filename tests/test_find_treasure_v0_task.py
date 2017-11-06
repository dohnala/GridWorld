import unittest

from env.env import GridWorldEnv
from env.state import Treasure


class FindTreasureTaskTest(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv("find_treasure_v0")
        self.env.state.agent.x = 0
        self.env.state.agent.y = 0

    def test_start_state(self):
        start_state = self.env.get_current_state()

        self.assertEqual(4, start_state.width)
        self.assertEqual(4, start_state.height)

        self.assertEqual(0, start_state.agent.x)
        self.assertEqual(0, start_state.agent.y)

        treasure = start_state.get_object_by_type(Treasure)

        self.assertEqual(2, treasure.x)
        self.assertEqual(3, treasure.y)

    def test_get_actions(self):
        self.assertListEqual([0, 1, 2, 3], self.env.get_actions())

    def test_reset(self):
        self.env.step(0)
        self.env.reset()

        state = self.env.get_current_state()

        self.assertEqual(0, state.step)

    def test_positive_goal(self):
        # (0, 0)

        reward, _, done = self.env.step(0)
        self.assertFalse(done)
        self.assertEqual(-0.01, reward)
        # (0, 1)

        reward, _, done = self.env.step(2)
        self.assertFalse(done)
        self.assertEqual(-0.01, reward)
        # (1, 1)

        reward, _, done = self.env.step(0)
        self.assertFalse(done)
        self.assertEqual(-0.01, reward)
        # (1, 2)

        reward, _, done = self.env.step(2)
        self.assertFalse(done)
        self.assertEqual(-0.01, reward)
        # (2, 2)

        reward, next_state, done = self.env.step(0)
        self.assertTrue(done)
        self.assertEqual(1, reward)
        self.assertEqual(5, next_state.step)
        self.assertEqual(2, next_state.agent.x)
        self.assertEqual(3, next_state.agent.y)

    def test_negative_goal(self):
        reward, next_state, done = None, None, None

        for step in range(10):
            self.env.step(0)
            reward, next_state, done = self.env.step(1)

        self.assertTrue(done)
        self.assertEqual(-0.01, reward)
        self.assertEqual(20, next_state.step)
        self.assertEqual(0, next_state.agent.x)
        self.assertEqual(0, next_state.agent.y)
