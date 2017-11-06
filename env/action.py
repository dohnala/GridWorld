class GridWorldAction:
    def __init__(self, name):
        """
        Initialize action with given name.

        :param name: action name
        """
        self.name = name

    def apply_if_valid(self, state):
        """
        Apply action to given state if si valid and return next state.

        :param state: state
        :return: next state
        """
        if self.is_valid(state):
            copy = state.copy()
            return self.apply(copy)
        else:
            return state

    def is_valid(self, state):
        """
        Return if action is valid in given state.

        :param state: state
        :return: if action is valid in given state
        """
        return False

    def apply(self, state):
        """
        Apply action to given state and return next state.

        :param state: state
        :return: next state
        """
        state.next_step()

        return state

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))


class MoveUp(GridWorldAction):
    """
    Move up agent in world.
    """

    def __init__(self):
        super().__init__("MoveUp")

    def is_valid(self, state):
        return state.agent.y < state.height - 1

    def apply(self, state):
        next_state = super().apply(state)
        next_state.agent.y += 1

        return next_state


class MoveDown(GridWorldAction):
    """
    Move down agent in world.
    """

    def __init__(self):
        super().__init__("MoveDown")

    def is_valid(self, state):
        return state.agent.y > 0

    def apply(self, state):
        next_state = super().apply(state)
        next_state.agent.y -= 1

        return next_state


class MoveRight(GridWorldAction):
    """
    Move right agent in world.
    """

    def __init__(self):
        super().__init__("MoveRight")

    def is_valid(self, state):
        return state.agent.x < state.width - 1

    def apply(self, state):
        next_state = super().apply(state)
        next_state.agent.x += 1

        return next_state


class MoveLeft(GridWorldAction):
    """
    Move left agent in world.
    """

    def __init__(self):
        super().__init__("MoveLeft")

    def is_valid(self, state):
        return state.agent.x > 0

    def apply(self, state):
        next_state = super().apply(state)
        next_state.agent.x -= 1

        return next_state
