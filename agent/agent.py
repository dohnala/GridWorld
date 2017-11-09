from enum import Enum

import torch


class EpisodeResult:
    """
    Result of one episode.
    """

    def __init__(self):
        self.episode = None
        self.reward = None
        self.steps = None
        self.has_won = None
        self.loss = None


class RunPhase(Enum):
    """
    Mark a running phase which agent can be in.
    """
    TRAIN = 0
    EVAL = 1


class Agent:
    """
    Agent interacting with an environment which can be trained or evaluated.
    """

    def __init__(self, name, env, model):
        self.name = name
        self.env = env
        self.model = model

    def train(self, num_episodes, result_writer):
        """
        Train agent for given number of episodes and write results.

        :param num_episodes: number of episodes to train
        :param result_writer: writer to write episode results into
        :return: None
        """
        # Set model to train mode
        self.model.train()

        for episode in range(num_episodes):
            result = self.__episode__(episode, RunPhase.TRAIN)
            result_writer.add_result(result)

    def eval(self, num_episodes, result_writer):
        """
        Evaluate agent for given number of episodes and write results.

        :param num_episodes: number of episodes to evaluate
        :param result_writer: writer to write episode results into
        :return: None
        """
        # Set model to eval mode
        self.model.eval()

        for episode in range(num_episodes):
            result = self.__episode__(episode, RunPhase.EVAL)
            result_writer.add_result(result)

    def save(self, file):
        """
        Save the model to given file.

        :param file: file
        :return: None
        """
        torch.save({'model': self.model.state_dict()}, file)

    def load(self, file):
        """
        Load the model from given file.

        :param file: file
        :return: None
        """
        self.model.load_state_dict(torch.load(file)['model'])

    def __episode__(self, episode, phase):
        """
        Play one episode and return result.

        :param episode: current episode
        :param phase: current phase
        :return: episode result
        """
        self.env.reset()

        reward = 0

        while not self.env.is_terminal():
            reward += self.__step__(phase)

        steps = self.env.state.step
        has_won = self.env.has_won()

        # Create an episode result and fill it
        result = EpisodeResult()
        result.episode = episode
        result.reward = reward
        result.steps = steps
        result.has_won = has_won

        self.__after_episode__(episode, phase, result)

        return result

    def __step__(self, phase):
        """
        Take one step on the environment and return obtained reward.

        :param phase: current phase
        :return: reward
        """
        state = self.env.state
        action = self.__select_action__(state, phase)
        reward, next_state, done = self.env.step(action)

        if phase == RunPhase.TRAIN:
            # Observer transition only in TRAIN phase
            self.__observe_transition__(state, action, reward, next_state, done)

        return reward

    def __select_action__(self, state, phase):
        """
        Select an action to take for given state.

        :param state: current state
        :param phase: current phase
        :return: action to take
        """
        pass

    def __observe_transition__(self, state, action, reward, next_state, done):
        """
        Observer current transition.

        :param state: state in which action was taken
        :param action: action taken
        :param reward: reward obtained for given action
        :param next_state: state action result in
        :param done: if next_state is terminal
        :return None
        """
        pass

    def __after_episode__(self, episode, phase, result):
        """
        Called after an episode is finished.

        :param episode: current episode
        :param phase: current phase
        :param result: episode result
        :return: None
        """
        pass
