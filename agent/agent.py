class EpisodeResult:
    """
    Result of one episode.
    """

    def __init__(self, episode, reward, steps, has_won):
        self.episode = episode
        self.reward = reward
        self.steps = steps
        self.has_won = has_won


class Agent:
    """
    Agent interacting with an environment which can be trained or evaluated.
    """

    def __init__(self, name, env):
        self.name = name
        self.env = env

    def train(self, num_episodes, result_writer):
        """
        Train agent for given number of episodes and write results.

        :param num_episodes: number of episodes to train
        :param result_writer: writer to write episode results into
        :return:
        """
        for episode in range(num_episodes):
            result = self.__episode__(episode)
            result_writer.add_result(result)

    def eval(self, num_episodes, result_writer):
        """
        Evaluate agent for given number of episodes and write results.

        :param num_episodes: number of episodes to evaluate
        :param result_writer: writer to write episode results into
        :return:
        """
        for episode in range(num_episodes):
            result = self.__episode__(episode)
            result_writer.add_result(result)

    def __episode__(self, episode):
        """
        Play one episode and return result.

        :param episode: current episode
        :return: episode result
        """
        self.env.reset()

        reward = 0

        while not self.env.is_terminal():
            reward += self.__step__()

        steps = self.env.state.step
        has_won = self.env.has_won()

        return EpisodeResult(episode, reward, steps, has_won)

    def __step__(self):
        """
        Take one step on the environment and return obtained reward.

        :return: reward
        """
        state = self.env.state
        action = self.__select_action__(state)
        reward, next_state, done = self.env.step(action)

        self.__observe_transition(state, action, reward, next_state, done)

        return reward

    def __select_action__(self, state):
        """
        Select an action to take for given state.

        :param state: current state
        :return: action to take
        """
        pass

    def __observe_transition(self, state, action, reward, next_state, done):
        """
        Observer current transition.

        :param state: state in which action was taken
        :param action: action taken
        :param reward: reward obtained for given action
        :param next_state: state action result in
        :param done: if next_state is terminal
        """
        pass
