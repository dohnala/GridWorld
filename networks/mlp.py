import torch.nn as nn
import torch.nn.functional as F

from networks import Network, NetworkModule


class MLP(Network):
    """
    Simple feed forward architecture composed of hidden layers.
    """

    def __init__(self, hidden_units=None):
        self.hidden_units = hidden_units

    def build(self, input_shape):
        return MLPModule(input_shape, self.hidden_units)

    def __str__(self):
        return "{}(hidden_units={})".format(self.__class__.__name__, self.hidden_units)


class MLPModule(NetworkModule):
    """
    Simple neural network composed of hidden layers.
    """

    def __init__(self, input_shape, hidden_units=None):
        super(MLPModule, self).__init__(input_shape)

        self.shape = input_shape

        # Create hidden layers
        layers = []
        if hidden_units:
            for h in hidden_units:
                layer = nn.Linear(self.shape, h)

                # Init parameters of hidden layer
                self.__init__parameters__(layer)

                layers.append(layer)

                self.shape = h

        self.hidden = nn.ModuleList(layers)

    def forward(self, states):
        result = states

        for layer in self.hidden:
            result = F.leaky_relu(layer(result))

        return result

    def output_shape(self):
        return self.shape

    @staticmethod
    def __init__parameters__(layer):
        nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant(layer.bias, 0)
