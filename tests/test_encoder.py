import unittest

import numpy as np
from numpy.testing import assert_array_equal

from agent.encoder import FeatureLayerEncoder, OneHotEncoder
from env.state import GridWorld, Agent, Treasure


class FeatureLayerTest(unittest.TestCase):

    def test_size_with_no_layers(self):
        encoder = FeatureLayerEncoder(4, 4, agent_position=False, treasure_position=False)

        self.assertEqual((4, 4, 0), encoder.size())

    def test_encode_state_with_no_layers(self):
        state = GridWorld(4, 4)

        encoder = FeatureLayerEncoder(4, 4, agent_position=False, treasure_position=False)

        assert_array_equal(np.empty((4, 4, 0)), encoder.encode(state))

    def test_size_with_agent_position_layer(self):
        encoder = FeatureLayerEncoder(4, 4, agent_position=True, treasure_position=False)

        self.assertEqual((4, 4, 1), encoder.size())

    def test_encode_state_with_agent_position_layer(self):
        state = GridWorld(4, 4)
        state.add_agent(Agent(0, 0))

        encoder = FeatureLayerEncoder(4, 4, agent_position=True, treasure_position=False)

        expected = [[[1, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]]

        assert_array_equal(expected, encoder.encode(state))

    def test_size_with_treasure_position_layer(self):
        encoder = FeatureLayerEncoder(4, 4, agent_position=False, treasure_position=True)

        self.assertEqual((4, 4, 1), encoder.size())

    def test_encode_with_treasure_position_layer(self):
        state = GridWorld(4, 4)
        state.add_object(Treasure(2, 3))

        encoder = FeatureLayerEncoder(4, 4, agent_position=False, treasure_position=True)

        expected = [[[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]]

        assert_array_equal(expected, encoder.encode(state))

    def test_size_with_multiple_layers(self):
        encoder = FeatureLayerEncoder(4, 4, agent_position=True, treasure_position=True)

        self.assertEqual((4, 4, 2), encoder.size())

    def test_encode_with_multiple_layers(self):
        state = GridWorld(4, 4)
        state.add_agent(Agent(0, 0))
        state.add_object(Treasure(2, 3))

        encoder = FeatureLayerEncoder(4, 4, agent_position=True, treasure_position=True)

        expected = [[[1, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]]

        assert_array_equal(expected, encoder.encode(state))


class OneHotEncoderTest(unittest.TestCase):

    def test_size_with_no_layers(self):
        encoder = OneHotEncoder(4, 4, agent_position=False, treasure_position=False)

        self.assertEqual(0, encoder.size())

    def test_encode_with_no_layers(self):
        state = GridWorld(4, 4)

        encoder = OneHotEncoder(4, 4, agent_position=False, treasure_position=False)

        assert_array_equal(np.empty(0), encoder.encode(state))

    def test_size_with_agent_position_layer(self):
        encoder = OneHotEncoder(4, 4, agent_position=True, treasure_position=False)

        self.assertEqual(16, encoder.size())

    def test_encode_with_agent_position_layer(self):
        state = GridWorld(4, 4)
        state.add_agent(Agent(0, 0))

        encoder = OneHotEncoder(4, 4, agent_position=True, treasure_position=False)

        expected = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert_array_equal(expected, encoder.encode(state))

    def test_size_with_treasure_position_layer(self):
        encoder = OneHotEncoder(4, 4, agent_position=False, treasure_position=True)

        self.assertEqual(16, encoder.size())

    def test_encode_with_treasure_position_layer(self):
        state = GridWorld(4, 4)
        state.add_object(Treasure(2, 3))

        encoder = OneHotEncoder(4, 4, agent_position=False, treasure_position=True)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        assert_array_equal(expected, encoder.encode(state))

    def test_size_with_multiple_layers(self):
        encoder = OneHotEncoder(4, 4, agent_position=True, treasure_position=True)

        self.assertEqual(32, encoder.size())

    def test_encode_with_multiple_layers(self):
        state = GridWorld(4, 4)
        state.add_agent(Agent(0, 0))
        state.add_object(Treasure(2, 3))

        encoder = OneHotEncoder(4, 4, agent_position=True, treasure_position=True)

        expected = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        assert_array_equal(expected, encoder.encode(state))
