from agents import Agent


class NStepAgent(Agent):
    """
    N-step agent which stores N transitions before model update
    """

    def __init__(self, name, env, encoder, model, optimizer, train_policy, eval_policy, n_step=1):
        super(NStepAgent, self).__init__(name, env, encoder, model, optimizer, train_policy, eval_policy)

        self.n_step = n_step
        self.transitions = []

    def __observe_transition__(self, state, action, reward, next_state, done):
        # Store transition
        self.transitions.append((self.__encode_state__(state),
                                 action,
                                 reward,
                                 self.__encode_state__(next_state),
                                 done))

        # Perform update when episode is finished or N transitions are gathered
        if done or len(self.transitions) == self.n_step:
            # Update model using given transitions and store loss
            self.last_loss = self.model.update(*self.__split_transitions__(self.transitions))

            # Clear transitions after an update
            self.transitions = []
