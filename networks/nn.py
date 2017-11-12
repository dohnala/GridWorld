import torch.nn as nn
import torch.nn.functional as F

from networks import Network, NetworkModule


class NN(Network):
    def __init__(self, hidden_units=None):
        self.hidden_units = hidden_units

    def build(self, input_shape):
        return NNModule(input_shape, self.hidden_units)


class NNModule(NetworkModule):
    def __init__(self, input_shape, hidden_units=None):
        super(NNModule, self).__init__(input_shape)

        self.shape = input_shape

        # Create hidden layers
        layers = []
        if hidden_units:
            for h in hidden_units:
                layers.append(nn.Linear(self.shape, h))
                self.shape = h

        self.hidden = nn.ModuleList(layers)

    def forward(self, states):
        result = states

        for layer in self.hidden:
            result = F.relu(layer(result))

        return result

    def output_shape(self):
        return self.shape
