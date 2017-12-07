import numpy as np

from policies import ExplorationPolicy


class EpsilonGreedyPolicy(ExplorationPolicy):
    """
    Epsilon greedy policy chooses random action with epsilon probability and
    the best action with 1 - epsilon probability.
    """

    def __init__(self, epsilon_initial, epsilon_final, epsilon_episodes):
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_episodes = epsilon_episodes
        self.epsilon_delta = (epsilon_initial - epsilon_final) / float(epsilon_episodes)
        self.epsilon = self.epsilon_initial

    def select_action(self, action_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(action_values))
        else:
            return np.argmax(action_values)

    def update(self):
        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_delta
        elif self.epsilon < self.epsilon_final:
            self.epsilon = self.epsilon_final

    def reset(self):
        self.epsilon = self.epsilon_initial

    def __str__(self):
        return "{}(epsilon_initial={}, epsilon_final={}, epsilon_episodes={})".format(
            self.__class__.__name__, self.epsilon_initial, self.epsilon_final, self.epsilon_episodes)
