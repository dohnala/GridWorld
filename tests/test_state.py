import unittest

from env.state import GridWorld, Agent, Treasure


class GridWorldTest(unittest.TestCase):
    def test_init(self):
        world = GridWorld(10, 10)

        self.assertEqual(0, world.step)
        self.assertEqual(10, world.width)
        self.assertEqual(10, world.height)

        self.assertIsNone(world.agent)
        self.assertEqual(0, len(world.get_objects()))
        self.assertEqual(0, len(world.get_object_types()))

    def test_add_agent(self):
        world = GridWorld(10, 10)
        world.add_agent(Agent(0, 0))

        self.assertEqual(0, world.agent.x)
        self.assertEqual(0, world.agent.y)

        self.assertEqual(1, len(world.get_objects()))
        self.assertListEqual([Agent], world.get_object_types())
        self.assertEqual(1, len(world.get_objects_by_type(Agent)))

    def test_add_treasure(self):
        world = GridWorld(10, 10)
        world.add_object(Treasure(8, 8))

        self.assertEqual(1, len(world.get_objects()))
        self.assertListEqual([Treasure], world.get_object_types())
        self.assertEqual(1, len(world.get_objects_by_type(Treasure)))

    def test_agent_is_at_treasure(self):
        world = GridWorld(10, 10)
        world.add_agent(Agent(0, 0))
        world.add_object(Treasure(8, 8))

        self.assertFalse(world.agent.is_at_any_object(world.get_objects_by_type(Treasure)))

        world.agent.x = 8
        world.agent.y = 8

        self.assertTrue(world.agent.is_at_any_object(world.get_objects_by_type(Treasure)))

    def test_next_step(self):
        world = GridWorld(10, 10)
        world.add_agent(Agent(0, 0))
        world.add_object(Treasure(8, 8))

        world.next_step()
        self.assertEqual(1, world.step)

    def test_copy(self):
        world = GridWorld(10, 10)
        world.add_agent(Agent(0, 0))
        world.add_object(Treasure(8, 8))

        copy = world.copy()
        copy.step = 1
        copy.width = 5
        copy.height = 5
        copy.agent.x = 1
        copy.agent.y = 2

        copy_treasure = copy.get_object_by_type(Treasure)

        copy_treasure.x = 2
        copy_treasure.y = 3

        self.assertEqual(0, world.step)
        self.assertEqual(10, world.width)
        self.assertEqual(10, world.height)

        self.assertEqual(0, world.agent.x)
        self.assertEqual(0, world.agent.y)

        treasure = world.get_object_by_type(Treasure)

        self.assertIsNotNone(treasure)
        self.assertEqual(8, treasure.x)
        self.assertEqual(8, treasure.y)

        self.assertEqual(1, copy.step)
        self.assertEqual(5, copy.width)
        self.assertEqual(5, copy.height)

        self.assertEqual(1, copy.agent.x)
        self.assertEqual(2, copy.agent.y)

        self.assertEqual(2, copy_treasure.x)
        self.assertEqual(3, copy_treasure.y)


class TreasureTest(unittest.TestCase):
    def test_init(self):
        treasure = Treasure(0, 0)

        self.assertEqual(0, treasure.x)
        self.assertEqual(0, treasure.y)

    def test_copy(self):
        treasure = Treasure(0, 0)

        copy = treasure.copy()
        copy.x = 10
        copy.y = 20

        self.assertEqual(0, treasure.x)
        self.assertEqual(0, treasure.y)

        self.assertEqual(10, copy.x)
        self.assertEqual(20, copy.y)


class AgentTest(unittest.TestCase):
    def test_init(self):
        agent = Agent(0, 0)

        self.assertEqual(0, agent.x)
        self.assertEqual(0, agent.y)

    def test_copy(self):
        agent = Agent(0, 0)

        copy = agent.copy()
        copy.x = 10
        copy.y = 20

        self.assertEqual(0, agent.x)
        self.assertEqual(0, agent.y)

        self.assertEqual(10, copy.x)
        self.assertEqual(20, copy.y)
