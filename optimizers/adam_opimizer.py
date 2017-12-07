import torch.optim as optim

from optimizers import Optimizer, OptimizerCreator


class AdamOptimizer(OptimizerCreator):
    """
    Adam optimizer implemented by PyTorch.
    """

    def __init__(self, learning_rate):
        super(AdamOptimizer, self).__init__()

        self.learning_rate = learning_rate

    def create(self, parameters):
        return Optimizer(optim.Adam(parameters, lr=self.learning_rate), parameters)

    def __str__(self):
        return "{}(learning_rate={})".format(self.__class__.__name__, self.learning_rate)
