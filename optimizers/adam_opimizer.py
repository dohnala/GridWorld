import torch.optim as optim

from optimizers import Optimizer


class AdamOptimizer(Optimizer):
    """
    Adam optimizer implemented by PyTorch.
    """

    def __init__(self, learning_rate):
        super(AdamOptimizer, self).__init__()

        self.learning_rate = learning_rate
        self.optimizer = None

    def step(self, loss, parameters):
        if self.optimizer is None:
            self.optimizer = optim.Adam(parameters, lr=self.learning_rate)

        # Zero all gradients
        self.optimizer.zero_grad()

        # Compute all gradients w.r.t given loss
        loss.backward()

        # Update all variables with computed gradients
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict() if self.optimizer else {}

    def load_state_dict(self, state_dict):
        if self.optimizer:
            self.optimizer.load_state_dict(state_dict)
