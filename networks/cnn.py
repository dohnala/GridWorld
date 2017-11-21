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


class CNNModule(NetworkModule):
    """
    Convolution neural network.
    """
    def __init__(self, input_shape, hidden_units=None):
        super(CNNModule, self).__init__(input_shape)

        width, height, num_layers = input_shape

        self.conv1 = nn.Conv2d(in_channels=num_layers, out_channels=16, kernel_size=3)
        self.nn = NNModule(784, hidden_units)

    def forward(self, states):
        result = F.relu(self.conv1(states))
        result = result.view(result.size(0), -1)
        result = self.nn(result)

        return result

    def output_shape(self):
        return self.nn.output_shape()
