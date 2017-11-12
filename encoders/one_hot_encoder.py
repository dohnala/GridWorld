from encoders import LayerEncoder


class OneHotEncoder(LayerEncoder):
    """
    Encodes a grid world as one hot vector.
    """

    def __init__(self, width, height, agent_position=True, treasure_position=False):
        super(OneHotEncoder, self).__init__(width, height, agent_position, treasure_position)

    def shape(self):
        """
        Return shape of encoded state as number.

        :return: shape of encoded state
        """
        width, height, num_layers = super().shape()

        return width * height * num_layers

    def encode(self, state):
        """
        Return encoded state as numpy array.

        :param state: state
        :return: encoded state
        """
        return super().encode(state).flatten()
