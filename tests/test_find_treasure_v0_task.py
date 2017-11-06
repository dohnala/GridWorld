import unittest

from env.action import MoveUp, MoveDown, MoveRight, MoveLeft
from env.env import GridWorldEnv
from env.state import GridWorld, Agent, Treasure


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
        self.assertListEqual([MoveUp(), MoveDown(), MoveRight(), MoveLeft()], self.env.get_actions())

    def test_reset(self):
        self.env.step(0)
        self.env.reset()

        state = self.env.get_current_state()

        self.assertEqual(0, state.step)

    def test_positive_goal(self):
        # (0, 0)

        _, reward = self.env.step(0)
        self.assertFalse(self.env.is_terminal())
        self.assertEqual(-0.01, reward)
        # (0, 1)

        _, reward = self.env.step(2)
        self.assertFalse(self.env.is_terminal())
        self.assertEqual(-0.01, reward)
        # (1, 1)

        _, reward = self.env.step(0)
        self.assertFalse(self.env.is_terminal())
        self.assertEqual(-0.01, reward)
        # (1, 2)

        _, reward = self.env.step(2)
        self.assertFalse(self.env.is_terminal())
        self.assertEqual(-0.01, reward)
        # (2, 2)

        next_state, reward = self.env.step(0)
        self.assertTrue(self.env.is_terminal())
        self.assertEqual(1, reward)
        self.assertEqual(5, next_state.step)
        self.assertEqual(2, next_state.agent.x)
        self.assertEqual(3, next_state.agent.y)

    def test_negative_goal(self):
        next_state, reward = None, None

        for step in range(10):
            self.env.step(0)
            next_state, reward = self.env.step(1)

        self.assertTrue(self.env.is_terminal())
        self.assertEqual(-0.01, reward)
        self.assertEqual(20, next_state.step)
        self.assertEqual(0, next_state.agent.x)
        self.assertEqual(0, next_state.agent.y)
