class ExplorationPolicy:
    """
    Exploration policy used to choose actions.
    """

    def select_action(self, action_values, current_step):
        """
        Given a list of values corresponding to each action, select one action according to the exploration policy.

        :param action_values: list of action values
        :param current_step: current step
        :return: chosen action
        """
        pass

    def __str__(self):
        return self.__class__.__name__
