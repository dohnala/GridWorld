import numpy as np

from agents import Agent
from models import QNstepModel
from policies import GreedyPolicy


class DQNAgent(Agent):
    """
    Deep-Q-Network agent.
    """

    def __init__(self, env, encoder, optimizer, discount, exploration_policy, n_step=1, hidden_units=None):
        # Create model
        model = QNstepModel(encoder.shape(),
                            env.num_actions,
                            discount,
                            hidden_units)

        super(DQNAgent, self).__init__("DQN agent", env, encoder, model, optimizer, exploration_policy, GreedyPolicy())

        self.n_step = n_step
        self.transitions = []

    def __observe_transition__(self, state, action, reward, next_state, done):
        # Store transition
        self.transitions.append((self.__encode_state__(state), action, reward, self.__encode_state__(next_state), done))

        # Perform update when episode is finished or N transitions are gathered
        if done or len(self.transitions) == self.n_step:
            transitions = np.array(self.transitions)
            states = np.vstack(transitions[:, 0])
            actions = np.vstack(transitions[:, 1])
            rewards = np.vstack(transitions[:, 2])
            next_states = np.vstack(transitions[:, 3])
            done = np.vstack(transitions[:, 4])

            # Update model using given experience and store loss
            self.last_loss = self.model.update(states, actions, rewards, next_states, done)

            # Clear transitions after an update
            self.transitions = []
