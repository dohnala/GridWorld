import torch.nn as nn


class Network:
    """
    Network representing neural network architecture.
    """
    def build(self, input_shape):
        """
        Build network using given architecture.

        :param input_shape: input shape
        :return: network
        """
        pass

    def __str__(self):
        return self.__class__.__name__


class NetworkModule(nn.Module):
    """
    Neural network implemented by PyTorch.
    """
    def __init__(self, input_shape):
        super(NetworkModule, self).__init__()

        self.input_shape = input_shape

    def forward(self, states):
        """
        Compute forward pass of the network for given states and return result.

        :param states: states
        :return: result
        """
        pass

    def output_shape(self):
        """
        Return shape of output layer of this network.

        :return: shape of output layer
        """
        pass
