import unittest

import numpy as np
from numpy.testing import assert_array_equal

from agent.encoder import FeatureLayerEncoder, OneHotEncoder
from env.env import GridWorldEnv


class FeatureLayerTest(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv("find_treasure_v0")
        self.env.state.agent.x = 0
        self.env.state.agent.y = 0

    def test_size_with_no_layers(self):
        encoder = FeatureLayerEncoder(self.env, agent_position=False, treasure_position=False)

        self.assertEqual((4, 4, 0), encoder.size())

    def test_encode_with_no_layers(self):
        encoder = FeatureLayerEncoder(self.env, agent_position=False, treasure_position=False)

        encoded_state = encoder.encode(self.env.get_current_state())

        assert_array_equal(np.empty((4, 4, 0)), encoded_state)

    def test_size_with_agent_position_layer(self):
        encoder = FeatureLayerEncoder(self.env, agent_position=True, treasure_position=False)

        self.assertEqual((4, 4, 1), encoder.size())

    def test_encode_with_agent_position_layer(self):
        encoder = FeatureLayerEncoder(self.env, agent_position=True, treasure_position=False)

        encoded_state = encoder.encode(self.env.get_current_state())

        expected = [[[1, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]]]

        assert_array_equal(expected, encoded_state)

    def test_size_with_treasure_position_layer(self):
        encoder = FeatureLayerEncoder(self.env, agent_position=False, treasure_position=True)

        self.assertEqual((4, 4, 1), encoder.size())

    def test_encode_with_treasure_position_layer(self):
        encoder = FeatureLayerEncoder(self.env, agent_position=False, treasure_position=True)

        encoded_state = encoder.encode(self.env.get_current_state())

        expected = [[[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]]

        assert_array_equal(expected, encoded_state)

    def test_size_with_multiple_layers(self):
        encoder = FeatureLayerEncoder(self.env, agent_position=True, treasure_position=True)

        self.assertEqual((4, 4, 2), encoder.size())

    def test_encode_with_multiple_layers(self):
        encoder = FeatureLayerEncoder(self.env, agent_position=True, treasure_position=True)

        encoded_state = encoder.encode(self.env.get_current_state())

        expected = [[[1, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]]]

        assert_array_equal(expected, encoded_state)


class OneHotEncoderTest(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv("find_treasure_v0")
        self.env.state.agent.x = 0
        self.env.state.agent.y = 0

    def test_size_with_no_layers(self):
        encoder = OneHotEncoder(self.env, agent_position=False, treasure_position=False)

        self.assertEqual(0, encoder.size())

    def test_encode_with_no_layers(self):
        encoder = OneHotEncoder(self.env, agent_position=False, treasure_position=False)

        encoded_state = encoder.encode(self.env.get_current_state())

        assert_array_equal(np.empty(0), encoded_state)

    def test_size_with_agent_position_layer(self):
        encoder = OneHotEncoder(self.env, agent_position=True, treasure_position=False)

        self.assertEqual(16, encoder.size())

    def test_encode_with_agent_position_layer(self):
        encoder = OneHotEncoder(self.env, agent_position=True, treasure_position=False)

        encoded_state = encoder.encode(self.env.get_current_state())

        expected = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert_array_equal(expected, encoded_state)

    def test_size_with_treasure_position_layer(self):
        encoder = OneHotEncoder(self.env, agent_position=False, treasure_position=True)

        self.assertEqual(16, encoder.size())

    def test_encode_with_treasure_position_layer(self):
        encoder = OneHotEncoder(self.env, agent_position=False, treasure_position=True)

        encoded_state = encoder.encode(self.env.get_current_state())

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        assert_array_equal(expected, encoded_state)

    def test_size_with_multiple_layers(self):
        encoder = OneHotEncoder(self.env, agent_position=True, treasure_position=True)

        self.assertEqual(32, encoder.size())

    def test_encode_with_multiple_layers(self):
        encoder = OneHotEncoder(self.env, agent_position=True, treasure_position=True)

        encoded_state = encoder.encode(self.env.get_current_state())

        expected = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

        assert_array_equal(expected, encoded_state)
