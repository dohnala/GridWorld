import unittest

from env.env import GridWorldEnv
from env.state import Treasure
from env.tasks import FindTreasureTaskV1


class FindTreasureTaskTest(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv(FindTreasureTaskV1(), seed=1)

    def test_start_state(self):
        start_state = self.env.state

        self.assertEqual(9, start_state.width)
        self.assertEqual(9, start_state.height)

        self.assertEqual(4, start_state.agent.x)
        self.assertEqual(1, start_state.agent.y)

        treasure = start_state.get_object_by_type(Treasure)

        self.assertEqual(2, treasure.x)
        self.assertEqual(1, treasure.y)

    def test_get_actions(self):
        self.assertListEqual([0, 1, 2, 3], self.env.actions)

    def test_reset(self):
        self.env.step(0)
        self.env.reset()

        state = self.env.state

        self.assertEqual(0, state.step)

    def test_positive_goal(self):
        # (4, 1)

        reward, _, done = self.env.step(3)
        self.assertFalse(done)
        self.assertEqual(-0.01, reward)
        # (3, 1)

        reward, next_state, done = self.env.step(3)
        self.assertTrue(done)
        self.assertEqual(1, reward)
        self.assertEqual(2, next_state.step)
        self.assertEqual(2, next_state.agent.x)
        self.assertEqual(1, next_state.agent.y)

    def test_negative_goal(self):
        reward, next_state, done = None, None, None

        for step in range(30):
            self.env.step(0)
            reward, next_state, done = self.env.step(1)

        self.assertTrue(done)
        self.assertEqual(-0.01, reward)
        self.assertEqual(60, next_state.step)
        self.assertEqual(4, next_state.agent.x)
        self.assertEqual(1, next_state.agent.y)
