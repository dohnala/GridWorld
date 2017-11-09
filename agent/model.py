import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module):
    """
    Model used for selection an action from given state.
    """

    def __init__(self, encoder, num_actions):
        """
        Initialize model.

        :param encoder: encoder used to encode states.
        :param num_actions: number of actions
        """
        super(Model, self).__init__()

        self.encoder = encoder
        self.num_actions = num_actions

    def forward(self, states):
        pass


class NNModel(Model):
    """
    Model implemented by feed forward neural network.
    """

    def __init__(self, encoder, num_actions, hidden_units=None):
        """
        Initialize model.

        :param encoder: one-hot encoder used to encode states.
        :param num_actions: number of actions
        :param hidden_units: list of hidden units
        """
        super(NNModel, self).__init__(encoder, num_actions)

        input_size = encoder.size()

        # Create hidden layers
        layers = []
        if hidden_units:
            for h in hidden_units:
                layers.append(nn.Linear(input_size, h))
                input_size = h

        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(input_size, num_actions)

    def forward(self, states):
        """
        Return action values computed by forward pass and return them as numpy array.

        :param states: list of states
        :return: action values for given states
        """
        # Encode given states using encoder and turn them into variable
        x = np.array(list(map(lambda state: self.encoder.encode(state), states)))
        x = Variable(torch.from_numpy(x))

        result = x

        for layer in self.hidden:
            result = F.relu(layer(result))

        result = self.output(result)

        return result
