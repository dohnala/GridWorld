class GridWorldTask:
    """
    Grid world task which acts like MDP.
    """

    def __init__(self, width, height, grid_world_generator):
        """
        Initialize task with given width, height and grid world generator.

        :param width: grid world width
        :param height: grid world height
        :param grid_world_generator: grid world generator
        """
        self.width = width
        self.height = height
        self.grid_world_generator = grid_world_generator

    def get_start_state(self):
        """
        Return starting state.

        :return: starting state
        """
        return self.grid_world_generator()

    def get_actions(self):
        """
        Return list of all actions.

        :return: list of all actions
        """
        return []

    def get_reward(self, state, action, next_state):
        """
        Return reward for applying given action to given state.

        :param state: state
        :param action: action
        :param next_state: next state
        :return:
        """
        return 0

    def is_terminal(self, state):
        """
        Return if given state is terminal or not.

        :param state: state
        :return: if given state is terminal or not
        """
        if self.is_winning(state) or self.is_losing(state):
            return True
        else:
            return False

    def is_winning(self, state):
        """
        Return if given state is winning or not.

        :param state: state
        :return: if given state is winning or not.
        """
        return False

    def is_losing(self, state):
        """
        Return if given state is losing or not.

        :param state: state
        :return: if given state is losing or not.
        """
        return False
