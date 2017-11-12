class Optimizer:
    """
    Optimizer which tries to update model parameters w.r.t some loss.
    """

    def set_parameters(self, parameters):
        """
        Set model parameters which should be updated.

        :param parameters: model parameters
        :return: None
        """
        pass

    def step(self, loss):
        """
        Perform optimization step by updating model parameters w.r.t given loss.

        :param loss: loss
        :return: None
        """
        pass

    def state_dict(self):
        """
        Return optimizer state as dict.

        :return: optimizer state as dict
        """
        pass

    def load_state_dict(self, state_dict):
        """
        Load given state dict to optimizer.

        :param state_dict: optimizer state dict
        :return: None
        """
        pass
