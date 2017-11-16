from agents import Agent, AgentConfig


class NStepAgentConfig(AgentConfig):
    """
    N-step agent's configuration.
    """

    def __init__(self, encoder, optimizer, train_policy, eval_policy, n_step):
        """
        Initialize configuration.

        :param encoder: encoder used to encode states
        :param optimizer: optimizer used to update model parameters
        :param train_policy: policy used in training phase
        :param eval_policy: policy used in evaluation phase
        :param n_step: how many steps are stored before updating the model
        """
        super(NStepAgentConfig, self).__init__(encoder, optimizer, train_policy, eval_policy)

        self.n_step = n_step


class NStepAgent(Agent):
    """
    N-step agent which stores N transitions before model update.
    """

    def __init__(self, name, model, config):
        """
        Initialize agent.

        :param name: name of the agent
        :param model: model used for action selection and learning
        :param config: agent's configuration
        """
        super(NStepAgent, self).__init__(name, model, config)

        self.n_step = config.n_step
        self.transitions = []

    def __observe_transition__(self, transition):
        # Store transition
        self.transitions.append(transition)

        # Perform update when episode is finished or N transitions are gathered
        if transition.done or len(self.transitions) == self.n_step:
            # Update model using given transitions and store loss
            self.last_loss = self.model.update(*self.__split_transitions__(self.transitions))

            # Clear transitions after an update
            self.transitions = []
