import torch.nn as nn


class ModelConfig:
    """
    Model's configuration.
    """

    def __init__(self, network):
        """
        Initialize configuration.

        :param network: network
        """
        self.network = network


class Model(nn.Module):
    """
    Model used for action selection and learning.
    """

    def __init__(self, input_shape, num_actions, config):
        """
        Initialize model.

        :param input_shape: shape of input state
        :param num_actions: number of actions
        :param config: model's config
        """
        super(Model, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.network = config.network.build(input_shape)
        self.optimizer = None

    def set_optimizer(self, optimizer):
        """
        Set optimizer used to update model parameters.

        :param optimizer: optimizer
        :return: None
        """
        self.optimizer = optimizer

    def forward(self, states):
        """
        Compute forward pass of the model for given states and return result.

        :param states: states
        :return: result
        """
        pass

    def update(self, states, actions, rewards, next_states, done):
        """
        Update model parameters using given experience.

        :param states: states
        :param actions: actions taken from states
        :param rewards: reward obtained by taking actions
        :param next_states: states resulting by taking actions
        :param done: flags representing if next states are terminals
        :return:
        """
        pass
