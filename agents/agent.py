from enum import Enum

import numpy as np
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


class AgentConfig:
    """
    Agent's configuration.
    """

    def __init__(self, encoder, optimizer, train_policy, eval_policy):
        """
        Initialize configuration.

        :param encoder: encoder used to encode states
        :param optimizer: optimizer used to update model parameters
        :param train_policy: policy used in training phase
        :param eval_policy: policy used in evaluation phase
        """
        self.encoder = encoder
        self.optimizer = optimizer
        self.train_policy = train_policy
        self.eval_policy = eval_policy


class Agent:
    """
    Agent interacting with an environment which can be trained or evaluated.
    """

    def __init__(self, name, env, model, config):
        """
        Initialize agent.

        :param name: name of the agent
        :param env: environment this agent interacts with
        :param model: model used for action selection and learning
        :param config: agent's configuration
        """
        self.name = name
        self.env = env
        self.model = model
        self.encoder = config.encoder
        self.optimizer = config.optimizer
        self.train_policy = config.train_policy
        self.eval_policy = config.eval_policy

        # Configure optimizer with model parameters
        self.optimizer.set_parameters(self.model.parameters())

        # Configure model with optimizer
        self.model.set_optimizer(self.optimizer)

        self.last_loss = None

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
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, file)

    def load(self, file):
        """
        Load the model from given file.

        :param file: file
        :return: None
        """
        checkpoint = torch.load(file)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

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
        result.loss = self.last_loss

        if phase == RunPhase.TRAIN:
            # Update policy after each training episode
            self.train_policy.update()

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
            # Observe transition only in TRAIN phase
            self.__observe_transition__(state, action, reward, next_state, done)

        return reward

    def __select_action__(self, state, phase):
        """
        Select an action to take for given state.

        :param state: current state
        :param phase: current phase
        :return: action to take
        """
        # Encode state and use model to predict action values
        action_values = self.model(self.__encode_state__(state)).data.numpy()

        # Select an action using policy
        if phase == RunPhase.TRAIN:
            return self.train_policy.select_action(action_values)

        if phase == RunPhase.EVAL:
            return self.eval_policy.select_action(action_values)

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
        # Create a transition
        transition = (self.__encode_state__(state), action, reward, self.__encode_state__(next_state), done)

        # Update model using given transition and store loss
        self.last_loss = self.model.update(*self.__split_transitions__([transition]))

    def __encode_state__(self, state):
        """
        Encode given state by configured encoders.

        :param state: state
        :return: encoded state
        """
        return self.encoder.encode(state)

    @staticmethod
    def __split_transitions__(transitions):
        """
        Split given transitions into batches of states, actions, rewards, next_states and done.

        :param transitions: transitions
        :return: tuple (states, actions, rewards, next_states, done)
        """
        transitions = np.array(transitions)

        states = np.vstack(transitions[:, 0])
        actions = np.vstack(transitions[:, 1])
        rewards = np.vstack(transitions[:, 2])
        next_states = np.vstack(transitions[:, 3])
        done = np.vstack(transitions[:, 4])

        return states, actions, rewards, next_states, done
