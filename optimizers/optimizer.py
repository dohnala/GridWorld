class OptimizerCreator:
    def create(self, parameters):
        pass


class Optimizer:
    """
    Optimizer which tries to update model parameters w.r.t some loss.
    """

    def __init__(self, optimizer, optimizer_parameters):
        """
        Initialize optimizer.

        :param optimizer: optimizer
        :param optimizer_parameters: optimizer's parameters
        """
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters

    def step(self, loss, parameters):
        """
        Perform optimization step by updating model parameters w.r.t given loss.

        :param loss: loss
        :param parameters: model parameters which should be updated
        :return: None
        """
        # Zero all gradients
        self.optimizer.zero_grad()

        # Compute all gradients w.r.t given loss
        loss.backward()

        # Ensure that given parameters share grads with optimizer parameters
        self.__ensure_shared_grads__(parameters)

        # Update all variables with computed gradients
        self.optimizer.step()

    def state_dict(self):
        """
        Return optimizer state as dict.

        :return: optimizer state as dict
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load given state dict to optimizer.

        :param state_dict: optimizer state dict
        :return: None
        """
        self.optimizer.load_state_dict(state_dict)

    def __ensure_shared_grads__(self, parameters):
        """
        Make sure that given parameters share grad with optimizer parameters.

        :param parameters: parameters
        :return: None
        """
        for param, shared_param in zip(parameters, self.optimizer_parameters):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
