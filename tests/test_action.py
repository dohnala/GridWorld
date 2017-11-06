import unittest

from env.action import MoveUp, MoveDown, MoveRight, MoveLeft
from env.state import GridWorld, Agent


class MoveUpTest(unittest.TestCase):
    def test_is_valid(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        self.assertTrue(MoveUp().is_valid(state))

    def test_is_valid_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 9))

        self.assertFalse(MoveUp().is_valid(state))

    def test_apply(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        next_state = MoveUp().apply_if_valid(state)

        self.assertEqual(1, next_state.agent.y)

    def test_apply_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 9))

        next_state = MoveUp().apply_if_valid(state)

        self.assertEqual(9, next_state.agent.y)


class MoveDownTest(unittest.TestCase):
    def test_is_valid(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 5))

        self.assertTrue(MoveDown().is_valid(state))

    def test_is_valid_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        self.assertFalse(MoveDown().is_valid(state))

    def test_apply(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 5))

        next_state = MoveDown().apply_if_valid(state)

        self.assertEqual(4, next_state.agent.y)

    def test_apply_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        next_state = MoveDown().apply_if_valid(state)

        self.assertEqual(0, next_state.agent.y)


class MoveRightTest(unittest.TestCase):
    def test_is_valid(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        self.assertTrue(MoveRight().is_valid(state))

    def test_is_valid_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(9, 0))

        self.assertFalse(MoveRight().is_valid(state))

    def test_apply(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        next_state = MoveRight().apply_if_valid(state)

        self.assertEqual(1, next_state.agent.x)

    def test_apply_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(9, 0))

        next_state = MoveRight().apply_if_valid(state)

        self.assertEqual(9, next_state.agent.x)


class MoveLeftTest(unittest.TestCase):
    def test_is_valid(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(5, 0))

        self.assertTrue(MoveLeft().is_valid(state))

    def test_is_valid_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        self.assertFalse(MoveLeft().is_valid(state))

    def test_apply(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(5, 0))

        next_state = MoveLeft().apply_if_valid(state)

        self.assertEqual(4, next_state.agent.x)

    def test_apply_on_boundary(self):
        state = GridWorld(10, 10)
        state.add_agent(Agent(0, 0))

        next_state = MoveLeft().apply_if_valid(state)

        self.assertEqual(0, next_state.agent.x)
