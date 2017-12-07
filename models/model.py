class ModelConfig:
    """
    Model's configuration.
    """

    def __init__(self, base_network):
        """
        Initialize configuration.

        :param base_network: base network
        """
        self.base_network = base_network


class Model:
    """
    Model used for action selection and learning.
    """

    def __init__(self, network, config):
        """
        Initialize model.

        :param network: main network used by model
        :param config: model's config
        """
        self.network = network
        self.config = config
        self.optimizer = None
        self.train_mode = None

        self.set_train_mode()

    def set_optimizer(self, optimizer):
        """
        Set optimizer used to update model parameters.

        :param optimizer: optimizer
        :return: None
        """
        self.optimizer = optimizer

    def predict(self, states):
        """
        Predict values for given states

        :param states: states
        :return: values depending on concrete model
        """
        pass

    def update(self, states, actions, rewards, next_states, done):
        """
        Update model parameters using given experience.

        :param states: states
        :param actions: actions taken from states
        :param rewards: reward obtained by taking actions
        :param next_states: states resulting by taking actions
        :param done: flags representing if next states are terminals
        :return: None
        """
        pass

    def parameters(self):
        """
        Return all trainable parameters of this model.

        :return: all trainable parameters
        """
        return list(self.network.parameters())

    def share_memory(self):
        """
        Move model parameters to share memory.

        :return: None
        """
        self.network.share_memory()

    def state_dict(self):
        """
        Return dictionary representing state of this model.

        :return: dictionary representing state
        """
        return self.network.state_dict()

    def load_state_dict(self, state_dict):
        """
        Load given state dictionary.

        :param state_dict: state dictionary
        :return: None
        """
        self.network.load_state_dict(state_dict)

    def set_train_mode(self):
        """
        Set this model to train mode.

        :return: None
        """
        self.train_mode = True
        self.network.train()

    def is_train_mode(self):
        """
        Return True if model is in train mode.

        :return: True if model is in train mode
        """
        return self.train_mode

    def set_eval_mode(self):
        """
        Set this model to eval mode.

        :return: None
        """
        self.train_mode = False
        self.network.eval()

    def is_eval_mode(self):
        """
        Return True if model is in eval model.

        :return: True if model is in eval model
        """
        return not self.train_mode
