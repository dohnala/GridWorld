import numpy as np


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


class GreedyPolicy(ExplorationPolicy):
    """
    Greedy exploration policy always chooses the best action.
    """

    def select_action(self, action_values):
        return np.argmax(action_values)


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
