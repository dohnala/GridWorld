import unittest

import numpy as np
from numpy.testing import assert_array_equal

from encoders import LayerEncoder, OneHotEncoder
from env.state import GridWorld, Agent, Treasure


class FeatureLayerTest(unittest.TestCase):
    def test_shape_with_no_layers(self):
        encoder = LayerEncoder(4, 4, agent_position=False, treasure_position=False)

        self.assertEqual((4, 4, 0), encoder.shape())

    def test_encode_state_with_no_layers(self):
        state = GridWorld(4, 4)

        encoder = LayerEncoder(4, 4, agent_position=False, treasure_position=False)

        assert_array_equal(np.empty((4, 4, 0)), encoder.encode(state))

    def test_shape_with_agent_position_layer(self):
        encoder = LayerEncoder(4, 4, agent_position=True, treasure_position=False)

        self.assertEqual((4, 4, 1), encoder.shape())

    def test_encode_state_with_agent_position_layer(self):
        state = GridWorld(4, 4)
        state.add_agent(Agent(0, 0))

        encoder = LayerEncoder(4, 4, agent_position=True, treasure_position=False)

        expected = [[[1, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]]

        assert_array_equal(expected, encoder.encode(state))

    def test_shape_with_treasure_position_layer(self):
        encoder = LayerEncoder(4, 4, agent_position=False, treasure_position=True)

        self.assertEqual((4, 4, 1), encoder.shape())

    def test_encode_with_treasure_position_layer(self):
        state = GridWorld(4, 4)
        state.add_object(Treasure(2, 3))

        encoder = LayerEncoder(4, 4, agent_position=False, treasure_position=True)

        expected = [[[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]]

        assert_array_equal(expected, encoder.encode(state))

    def test_shape_with_multiple_layers(self):
        encoder = LayerEncoder(4, 4, agent_position=True, treasure_position=True)

        self.assertEqual((4, 4, 2), encoder.shape())

    def test_encode_with_multiple_layers(self):
        state = GridWorld(4, 4)
        state.add_agent(Agent(0, 0))
        state.add_object(Treasure(2, 3))

        encoder = LayerEncoder(4, 4, agent_position=True, treasure_position=True)

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
    def test_shape_with_no_layers(self):
        encoder = OneHotEncoder(4, 4, agent_position=False, treasure_position=False)

        self.assertEqual(0, encoder.shape())

    def test_encode_with_no_layers(self):
        state = GridWorld(4, 4)

        encoder = OneHotEncoder(4, 4, agent_position=False, treasure_position=False)

        assert_array_equal(np.empty(0), encoder.encode(state))

    def test_shape_with_agent_position_layer(self):
        encoder = OneHotEncoder(4, 4, agent_position=True, treasure_position=False)

        self.assertEqual(16, encoder.shape())

    def test_encode_with_agent_position_layer(self):
        state = GridWorld(4, 4)
        state.add_agent(Agent(0, 0))

        encoder = OneHotEncoder(4, 4, agent_position=True, treasure_position=False)

        expected = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert_array_equal(expected, encoder.encode(state))

    def test_shape_with_treasure_position_layer(self):
        encoder = OneHotEncoder(4, 4, agent_position=False, treasure_position=True)

        self.assertEqual(16, encoder.shape())

    def test_encode_with_treasure_position_layer(self):
        state = GridWorld(4, 4)
        state.add_object(Treasure(2, 3))

        encoder = OneHotEncoder(4, 4, agent_position=False, treasure_position=True)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        assert_array_equal(expected, encoder.encode(state))

    def test_shape_with_multiple_layers(self):
        encoder = OneHotEncoder(4, 4, agent_position=True, treasure_position=True)

        self.assertEqual(32, encoder.shape())

    def test_encode_with_multiple_layers(self):
        state = GridWorld(4, 4)
        state.add_agent(Agent(0, 0))
        state.add_object(Treasure(2, 3))

        encoder = OneHotEncoder(4, 4, agent_position=True, treasure_position=True)

        expected = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        assert_array_equal(expected, encoder.encode(state))
