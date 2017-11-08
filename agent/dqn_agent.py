import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from agent.agent import Agent, RunPhase
from agent.policy import GreedyPolicy


class DQNAgent(Agent):
    """
    Deep-Q-Network agent.
    """

    def __init__(self, env, model, learning_rate, discount, exploration_policy):
        """
        Initialize the agent.

        :param env: environment
        :param model: model for selecting actions
        :param learning_rate: learning rate
        :param discount: discount factor
        :param exploration_policy: exploration policy used during training
        """
        super(DQNAgent, self).__init__("DQN agent", env)

        self.model = model
        self.learning_rate = learning_rate
        self.discount = discount
        self.train_policy = exploration_policy
        self.eval_policy = GreedyPolicy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.current_loss = None

    def __select_action__(self, state, phase):
        # Run model to get action values and convert them to numpy array
        action_values = self.model(state).data.numpy()

        # Select action using policy
        if phase == RunPhase.TRAIN:
            return self.train_policy.select_action(action_values)

        if phase == RunPhase.EVAL:
            return self.eval_policy.select_action(action_values)

    def __observe_transition__(self, state, action, reward, next_state, done):
        # Compute target for given transition
        if done:
            # target = r
            target = reward
        else:
            # target = r + d * maxQ(s', a')
            max_value = self.model(next_state).data.numpy().max()
            target = reward + self.discount * max_value

        # Train the model using given target
        self.__train__(state, action, target)

    def __train__(self, state, action, target):
        # Find model prediction for given state and action
        prediction = self.model(state).gather(0, Variable(torch.from_numpy(np.array([action]))))

        # Compute Huber loss from prediction and target
        loss = F.smooth_l1_loss(prediction, Variable(torch.from_numpy(np.array([target], dtype=np.float32))))

        # Zero all gradients
        self.optimizer.zero_grad()

        # Compute all gradients w.r.t given loss
        loss.backward()

        # Update all variables with computed gradients
        self.optimizer.step()

        # Save the loss
        self.current_loss = loss.data.numpy()

    def __after_episode__(self, episode, phase, result):
        if phase == RunPhase.TRAIN:
            # Update policy after each training episode
            self.train_policy.update()
