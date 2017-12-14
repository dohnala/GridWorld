import numpy as np

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

        self.n_step = self.config.n_step + 1 if self.config.keep_last else self.config.n_step
        self.keep_last = self.config.keep_last

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def __observe__(self, states, actions, rewards, next_states, dones):
        # Store transitions
        self.states.append(np.asarray(states))
        self.actions.append(np.vstack(actions))
        self.rewards.append(np.vstack(rewards))
        self.next_states.append(np.asarray(next_states))
        self.dones.append(np.vstack(dones))

        # Perform update when N transitions are observed
        if len(self.states) == self.n_step:
            # Update model using given transitions
            self.__update_model__(
                states=np.asarray(self.states),
                actions=np.asarray(self.actions),
                rewards=np.asarray(self.rewards),
                next_states=np.asarray(self.next_states),
                dones=np.asarray(self.dones))

            if self.keep_last:
                # Keep last transition after an update
                self.states = [self.states[-1]]
                self.actions = [self.actions[-1]]
                self.rewards = [self.rewards[-1]]
                self.next_states = [self.next_states[-1]]
                self.dones = [self.dones[-1]]
            else:
                # Clear transitions after an update
                self.states = []
                self.actions = []
                self.rewards = []
                self.next_states = []
                self.dones = []
