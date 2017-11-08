import numpy as np

from env.state import Treasure


class GridWorldEncoder:
    """Encodes a grid world into suitable format used for agent's model."""

    def __init__(self, width, height, agent_position=True, treasure_position=False):
        self.width = width
        self.height = height
        self.agent_position = agent_position
        self.treasure_position = treasure_position

    def size(self):
        pass

    def encode(self, state):
        pass


class FeatureLayerEncoder(GridWorldEncoder):
    """
    Encodes a grid world as list of stacked feature layers.
    """

    def __init__(self, width, height, agent_position=True, treasure_position=False):
        super(FeatureLayerEncoder, self).__init__(width, height, agent_position, treasure_position)

        self.num_layers = self.__get_num_layers__()

    def size(self):
        """
        Return size of encoded state as tuple (width, height, number of layers).

        :return: size of encoded state
        """
        return self.width, self.height, self.num_layers

    def encode(self, state):
        """
        Return encoded state as numpy array.

        :param state: state
        :return: encoded state
        """
        layers = self.__get_layers__(state)

        return np.stack(layers) if len(layers) > 0 else np.empty(self.size())

    def __get_layers__(self, state):
        layers = []

        if self.agent_position:
            layers.append(self.__create_agent_position_layer__(state))

        if self.treasure_position:
            layers.append(self.__create_treasure_position_layer__(state))

        return layers

    def __create_agent_position_layer__(self, state):
        agent = state.agent

        layer = self.__create_layer__()
        layer[agent.x][agent.y] = 1

        return layer

    def __create_treasure_position_layer__(self, state):
        layer = self.__create_layer__()

        for treasure in state.get_objects_by_type(Treasure):
            layer[treasure.x][treasure.y] = 1

        return layer

    def __create_layer__(self):
        return np.zeros((self.width, self.height), dtype=np.float32)

    def __get_num_layers__(self):
        num_layers = 0

        if self.agent_position:
            num_layers += 1

        if self.treasure_position:
            num_layers += 1

        return num_layers


class OneHotEncoder(FeatureLayerEncoder):
    """
    Encodes a grid world as one hot vector.
    """

    def __init__(self, width, height, agent_position=True, treasure_position=False):
        super(OneHotEncoder, self).__init__(width, height, agent_position, treasure_position)

    def size(self):
        """
        Return size of encoded state as number.

        :return: size of encoded state
        """
        width, height, num_layers = super().size()

        return width * height * num_layers

    def encode(self, state):
        """
        Return encoded state as numpy array.

        :param state: state
        :return: encoded state
        """
        return super().encode(state).flatten()
