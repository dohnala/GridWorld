class GridWorldAction:
    def apply(self, state):
        """
        Apply action to given state if si valid and return next state.

        :param state: state
        :return: next state
        """
        copy = state.copy()
        copy.next_step()

        if self.__is_valid__(state):
            return self.__apply__(copy)

        return copy

    def __is_valid__(self, state):
        """
        Return if action is valid in given state.

        :param state: state
        :return: if action is valid in given state
        """
        return False

    def __apply__(self, state):
        """
        Apply action to given state and return next state.

        :param state: state
        :return: next state
        """
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

    def __is_valid__(self, state):
        return state.agent.y < state.height - 1

    def __apply__(self, state):
        state.agent.y += 1

        return state


class MoveDown(GridWorldAction):
    """
    Move down agent in world.
    """

    def __is_valid__(self, state):
        return state.agent.y > 0

    def __apply__(self, state):
        state.agent.y -= 1

        return state


class MoveRight(GridWorldAction):
    """
    Move right agent in world.
    """

    def __is_valid__(self, state):
        return state.agent.x < state.width - 1

    def __apply__(self, state):
        state.agent.x += 1

        return state


class MoveLeft(GridWorldAction):
    """
    Move left agent in world.
    """

    def __is_valid__(self, state):
        return state.agent.x > 0

    def __apply__(self, state):
        state.agent.x -= 1

        return state
