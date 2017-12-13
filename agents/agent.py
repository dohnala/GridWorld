import collections
from enum import Enum

import numpy as np
import torch

from encoders import GridWorldEncoder
from models import Model
from optimizers import OptimizerCreator
from policies import ExplorationPolicy

Transition = collections.namedtuple('Transition', 'state action reward next_state done')


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
        assert isinstance(encoder, GridWorldEncoder), "encoder is not valid"
        assert isinstance(optimizer, OptimizerCreator), "optimizer is not valid"
        assert isinstance(train_policy, ExplorationPolicy), "train_policy is not valid"
        assert isinstance(eval_policy, ExplorationPolicy), "eval_policy is not valid"

        self.encoder = encoder
        self.optimizer = optimizer
        self.train_policy = train_policy
        self.eval_policy = eval_policy


class Agent:
    """
    Agent interacting with an environment which can be trained or evaluated.
    """

    def __init__(self, name, model, config):
        """
        Initialize agent.

        :param name: name of the agent
        :param model: model used for action selection and learning
        :param config: agent's configuration
        """
        assert type(name) is str, "name is not valid"
        assert isinstance(model, Model), "model is not valid"
        assert isinstance(config, AgentConfig), "config is not valid"

        self.name = name
        self.model = model
        self.config = config
        self.encoder = config.encoder
        self.train_policy = config.train_policy
        self.eval_policy = config.eval_policy

        # Current step
        self.step = 0

        # Current phase the agent is in
        self.current_phase = None

        # Currently used exploration policy
        self.current_policy = self.train_policy

        # Loss after last model update
        self.last_loss = None

        # Configure optimizer
        self.optimizer = self.__create_optimizer__(config.optimizer)

        # Configure model with optimizer
        self.model.set_optimizer(self.optimizer)

    def act(self, state, phase):
        """
        Select an action to take for given state and phase.

        :param state: current state
        :param phase: current phase
        :return: action to take
        """
        # Set phase
        if self.current_phase != phase:
            self.__set_phase__(phase)

        # Encode state and use model to predict action values
        states = np.expand_dims(self.__encode_state__(state), axis=0)
        action_values = self.model.predict(states)[0]

        # Select an action using policy
        action = self.current_policy.select_action(action_values, self.step)

        return action

    def observe(self, state, action, reward, next_state, done):
        """
        Observe current transition.

        :param state: state in which action was taken
        :param action: action taken
        :param reward: reward obtained for given action
        :param next_state: state action result in
        :param done: if next_state is terminal
        :return None
        """
        self.step += 1

        transition = Transition(
            state=self.__encode_state__(state),
            action=action,
            reward=reward,
            next_state=self.__encode_state__(next_state),
            done=done)

        self.__observe_transition__(transition)

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

    def __create_optimizer__(self, optimizer_creator):
        """
        Create optimizer.

        :param optimizer_creator: creator used to create optimizer
        :return: optimizer
        """
        return optimizer_creator.create(self.model.parameters())

    def __set_phase__(self, phase):
        """
        Change current phase.

        :param phase: new phase
        :return: None
        """
        # Set current phase
        self.current_phase = phase

        # Change model and policy for current phase
        if self.current_phase == RunPhase.TRAIN:
            self.model.set_train_mode()
            self.current_policy = self.train_policy

        if self.current_phase == RunPhase.EVAL:
            self.model.set_eval_mode()
            self.current_policy = self.eval_policy

    def __observe_transition__(self, transition):
        """
        Observe given transition.

        :param transition: transition
        :return: None
        """
        # Update model using given transition
        self.__update_model__([transition])

    def __update_model__(self, transitions):
        """
        Update model using given transitions.

        :param transitions: transitions
        :return: None
        """
        self.last_loss = self.model.update(*self.__split_transitions__(transitions))

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
        states, actions, rewards, next_states, done = zip(*transitions)

        states = np.asarray(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards).astype(np.float32)
        next_states = np.asarray(next_states)
        done = np.vstack(done).astype(np.uint8)

        return states, actions, rewards, next_states, done
