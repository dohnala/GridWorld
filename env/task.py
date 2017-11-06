class GridWorldTask:
    """
    Grid world task which acts like MDP.
    """
    def __init__(self, grid_world_generator):
        """
        Initialize task with given grid world generator.

        :param grid_world_generator: grid world generator
        """
        self.grid_world_generator = grid_world_generator

    def get_start_state(self):
        """
        Return starting state.

        :return: starting state
        """
        return self.grid_world_generator()

    def get_actions(self):
        """
        Return all actions.

        :return: all actions
        """
        return []

    def get_possible_actions(self, state):
        """
        Return actions which are valid in given state.

        :param state: state
        :return: actions
        """
        if self.is_terminal(state):
            return []

        return [action for action in self.get_actions() if action.is_valid(state)]

    def is_action_possible(self, state, action):
        """
        Return if given action is valid at given state.

        :param state: state
        :param action: action
        :return: if given action is valid at given state
        """
        return action in self.get_possible_actions(state)

    @staticmethod
    def apply_action(state, action):
        """
        Apply given action at given state.

        :param state: state
        :param action: action
        :return: next state
        """
        return action.apply_if_valid(state)

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
