import numpy as np

from policies import ExplorationPolicy


class GreedyPolicy(ExplorationPolicy):
    """
    Greedy exploration policy always chooses the best action.
    """

    def select_action(self, action_values):
        return np.argmax(action_values)
