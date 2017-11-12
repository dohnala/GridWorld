class GridWorldEncoder:
    """Encodes a grid world state into suitable format used for model."""

    def __init__(self, width, height, agent_position=True, treasure_position=False):
        self.width = width
        self.height = height
        self.agent_position = agent_position
        self.treasure_position = treasure_position

    def shape(self):
        """
        Return shape of encoded state.

        :return: shape of encoded state
        """
        pass

    def encode(self, state):
        """
        Encode state.

        :param state: state
        :return: encoded state
        """
        pass
