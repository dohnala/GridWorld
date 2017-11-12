import torch.nn as nn


class Network:
    def build(self, input_shape):
        pass


class NetworkModule(nn.Module):
    def __init__(self, input_shape):
        super(NetworkModule, self).__init__()

        self.input_shape = input_shape

    def forward(self, states):
        pass

    def output_shape(self):
        pass
