import unittest

from env.action import MoveUp, MoveDown, MoveRight, MoveLeft
from env.state import GridWorld, Agent


class MoveUpTest(unittest.TestCase):
    def test_is_valid(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        self.assertTrue(MoveUp().__is_valid__(state))

    def test_is_valid_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 9))

        self.assertFalse(MoveUp().__is_valid__(state))

    def test_apply(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        next_state = MoveUp().apply(state)

        self.assertEqual(1, next_state.step)
        self.assertEqual(1, next_state.agent.y)

    def test_apply_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 9))

        next_state = MoveUp().apply(state)

        self.assertEqual(1, next_state.step)
        self.assertEqual(9, next_state.agent.y)


class MoveDownTest(unittest.TestCase):
    def test_is_valid(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 5))

        self.assertTrue(MoveDown().__is_valid__(state))

    def test_is_valid_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        self.assertFalse(MoveDown().__is_valid__(state))

    def test_apply(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 5))

        next_state = MoveDown().apply(state)

        self.assertEqual(1, next_state.step)
        self.assertEqual(4, next_state.agent.y)

    def test_apply_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        next_state = MoveDown().apply(state)

        self.assertEqual(1, next_state.step)
        self.assertEqual(0, next_state.agent.y)


class MoveRightTest(unittest.TestCase):
    def test_is_valid(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        self.assertTrue(MoveRight().__is_valid__(state))

    def test_is_valid_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(9, 0))

        self.assertFalse(MoveRight().__is_valid__(state))

    def test_apply(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        next_state = MoveRight().apply(state)

        self.assertEqual(1, next_state.step)
        self.assertEqual(1, next_state.agent.x)

    def test_apply_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(9, 0))

        next_state = MoveRight().apply(state)

        self.assertEqual(1, next_state.step)
        self.assertEqual(9, next_state.agent.x)


class MoveLeftTest(unittest.TestCase):
    def test_is_valid(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(5, 0))

        self.assertTrue(MoveLeft().__is_valid__(state))

    def test_is_valid_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        self.assertFalse(MoveLeft().__is_valid__(state))

    def test_apply(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(5, 0))

        next_state = MoveLeft().apply(state)

        self.assertEqual(1, next_state.step)
        self.assertEqual(4, next_state.agent.x)

    def test_apply_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        next_state = MoveLeft().apply(state)

        self.assertEqual(1, next_state.step)
        self.assertEqual(0, next_state.agent.x)
