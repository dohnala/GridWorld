import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from networks import Network, NetworkModule
from networks.nn import NNModule


class CNN(Network):
    """
    Convolution neural network architecture.
    """
    def __init__(self, hidden_units=None):
        self.hidden_units = hidden_units

    def build(self, input_shape):
        return CNNModule(input_shape, self.hidden_units)

    def __str__(self):
        return "{}(hidden_units={})".format(self.__class__.__name__, self.hidden_units)


class CNNModule(NetworkModule):
    """
    Convolution neural network.
    """
    def __init__(self, input_shape, hidden_units=None):
        super(CNNModule, self).__init__(input_shape)

        width, height, num_layers = input_shape

        self.conv1 = nn.Conv2d(in_channels=num_layers, out_channels=16, kernel_size=3)

        # Init parameters of conv layer
        self.__init__parameters__(self.conv1)

        self.nn = NNModule(self.__conv_shape_flatten__(width, height, self.conv1), hidden_units)

    def forward(self, states):
        result = F.leaky_relu(self.conv1(states))
        result = result.view(result.size(0), -1)
        result = self.nn.forward(result)

        return result

    def output_shape(self):
        return self.nn.output_shape()

    @staticmethod
    def __init__parameters__(layer):
        nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant(layer.bias, 0)

    @staticmethod
    def __conv_shape_flatten__(w_in, h_in, conv):
        """
        Return flattened output shape of given convolution layer.

        :param w_in: input width
        :param h_in: input height
        :param conv: convolution layer
        :return: output shape of given convolution layer flattened
        """
        w_out = int((w_in + 2 * conv.padding[0] - conv.kernel_size[0]) / conv.stride[0]) + 1
        h_out = int((h_in + 2 * conv.padding[1] - conv.kernel_size[1]) / conv.stride[1]) + 1

        return w_out * h_out * conv.out_channels
