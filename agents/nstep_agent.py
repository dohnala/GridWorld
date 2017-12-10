from agents import Agent, AgentConfig


class NStepAgentConfig(AgentConfig):
    """
    N-step agent's configuration.
    """

    def __init__(self, encoder, optimizer, train_policy, eval_policy, n_step, keep_last=False):
        """
        Initialize configuration.

        :param encoder: encoder used to encode states
        :param optimizer: optimizer used to update model parameters
        :param train_policy: policy used in training phase
        :param eval_policy: policy used in evaluation phase
        :param n_step: how many steps are stored before updating the model
        :param keep_last: keep last transition after update
        """
        super(NStepAgentConfig, self).__init__(encoder, optimizer, train_policy, eval_policy)

        assert type(n_step) is int and n_step > 0, "n_step has to be integer greater than zero"
        assert type(keep_last) is bool, "keep_last has to be boolean"

        self.n_step = n_step
        self.keep_last = keep_last


class NStepAgent(Agent):
    """
    N-step agent which stores N transitions before model update.
    """

    def __init__(self, **kwargs):
        """
        Initialize agent.
        """
        super(NStepAgent, self).__init__(**kwargs)

        self.n_step = self.config.n_step
        self.keep_last = self.config.keep_last
        self.transitions = []

    def __observe_transition__(self, transition):
        # Store transition
        self.transitions.append(transition)

        # Perform update when episode is finished or N transitions are gathered
        if transition.done or len(self.transitions) == self.n_step:
            # Update model using given transitions
            self.__update_model__(self.transitions)

            if not transition.done and self.keep_last:
                # Keep last transition after an update
                self.transitions = [self.transitions[-1]]
            else:
                # Clear transitions after an update
                self.transitions = []
