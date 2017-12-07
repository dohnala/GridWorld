class ExplorationPolicy:
    """
    Exploration policy used to choose actions.
    """

    def select_action(self, action_values):
        """
        Given a list of values corresponding to each action, select one action according to the exploration policy.

        :param action_values: list of action values
        :return: chosen action
        """
        pass

    def update(self):
        """
        Used for updating the exploration policy parameters after episode is ended.

        :return: None
        """
        pass

    def reset(self):
        """
        Used for resetting the exploration policy parameters when needed.

        :return: None
        """
        pass

    def __str__(self):
        return self.__class__.__name__
