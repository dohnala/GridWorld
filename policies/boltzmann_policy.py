import numpy as np

from policies import ExplorationPolicy


class BoltzmannPolicy(ExplorationPolicy):
    """
    Boltzmann policy uses softmax function to turn action values into probabilities
    and use them to choose the action.
    """

    def __init__(self, temperature_initial, temperature_final, temperature_episodes):
        self.temperature_initial = temperature_initial
        self.temperature_final = temperature_final
        self.temperature_episodes = temperature_episodes
        self.temperature_delta = (temperature_initial - temperature_final) / float(temperature_episodes)
        self.temperature = self.temperature_initial

    def select_action(self, action_values):
        exp_probabilities = np.exp(np.array(action_values) / self.temperature)
        probabilities = exp_probabilities / np.sum(exp_probabilities)

        # Make sure probabilities sum to 1
        probabilities[-1] = 1 - np.sum(probabilities[:-1])

        # Choose action according to the probabilities
        return np.random.choice(range(len(action_values)), p=probabilities)

    def update(self):
        if self.temperature > self.temperature_final:
            self.temperature -= self.temperature_delta
        elif self.temperature < self.temperature_final:
            self.temperature = self.temperature_final

    def reset(self):
        self.temperature = self.temperature_initial
