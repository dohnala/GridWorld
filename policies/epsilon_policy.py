import numpy as np

from policies import ExplorationPolicy


class EpsilonGreedyPolicy(ExplorationPolicy):
    """
    Epsilon greedy policy chooses random action with epsilon probability and
    the best action with 1 - epsilon probability.
    """

    def __init__(self, epsilon_initial, epsilon_final, epsilon_steps):
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps

    def select_action(self, action_values, current_step):
        fraction = min(float(current_step) / self.epsilon_steps, 1.0)
        epsilon = self.epsilon_initial + fraction * (self.epsilon_final - self.epsilon_initial)

        if np.random.rand() < epsilon:
            return np.random.randint(len(action_values))
        else:
            return np.argmax(action_values)

    def __str__(self):
        return "{}(epsilon_initial={}, epsilon_final={}, epsilon_steps={})".format(
            self.__class__.__name__, self.epsilon_initial, self.epsilon_final, self.epsilon_steps)
