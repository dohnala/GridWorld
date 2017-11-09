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

    def __init__(self, env, model, learning_rate, discount, exploration_policy, n_step=1):
        """
        Initialize the agent.

        :param env: environment
        :param model: model for selecting actions
        :param learning_rate: learning rate
        :param discount: discount factor
        :param exploration_policy: exploration policy used during training
        .param n_step: how many steps to use to compute targets
        """
        super(DQNAgent, self).__init__("DQN agent", env)

        self.model = model
        self.learning_rate = learning_rate
        self.discount = discount
        self.train_policy = exploration_policy
        self.eval_policy = GreedyPolicy()
        self.n_step = 1
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.transitions = []
        self.last_loss = None

    def __select_action__(self, state, phase):
        # Run model to get action values and convert them to numpy array
        action_values = self.model([state]).data[0].numpy()

        # Select action using policy
        if phase == RunPhase.TRAIN:
            return self.train_policy.select_action(action_values)

        if phase == RunPhase.EVAL:
            return self.eval_policy.select_action(action_values)

    def __observe_transition__(self, state, action, reward, next_state, done):
        # Store transition
        self.transitions.append((state, action, reward, next_state, done))

        # Perform rollout when episode is finished or N transitions are gathered
        if done:
            # Value of terminal state is 0
            self.__rollout__(0)
        elif len(self.transitions) == self.n_step:
            # Value of next_state is maximum value across actions
            self.__rollout__(self.model([next_state]).data[0].numpy().max())

    def __rollout__(self, value):
        """
        Perform rollout on list of stored transitions and compute targets for training.

        :param value: value of current state
        :return: None
        """
        transitions = np.array(self.transitions)

        states = transitions[:, 0]
        actions = transitions[:, 1]
        rewards = transitions[:, 2]
        targets = np.zeros(len(rewards))

        target = value

        # Compute targets from rewards
        for i in reversed(range(len(rewards))):
            target = rewards[i] + self.discount * target
            targets[i] = target

        # Train the model using given targets
        self.__train__(states, actions, targets)

        # Clear transitions on the end of rollout
        self.transitions = []

    def __train__(self, states, actions, targets):
        """
        Train the model using given targets.

        :param states: states
        :param actions: actions
        :param targets: targets
        :return: None
        """
        # Compute model outputs for given states
        outputs = self.model(states)

        # Turn actions into variable
        actions = Variable(torch.from_numpy(np.array(actions, dtype=np.int64)))

        # Compute predictions of actions in given states
        predictions = outputs.gather(1, actions.unsqueeze(1)).squeeze()

        # Turn targets into variable
        targets = Variable(torch.from_numpy(np.array(targets, dtype=np.float32)))

        # Compute Huber loss from predictions and targets
        loss = F.smooth_l1_loss(predictions, targets)

        # Zero all gradients
        self.optimizer.zero_grad()

        # Compute all gradients w.r.t given loss
        loss.backward()

        # Update all variables with computed gradients
        self.optimizer.step()

        # Save the loss
        self.last_loss = loss.data.numpy()

    def __after_episode__(self, episode, phase, result):
        if phase == RunPhase.TRAIN:
            # Update policy after each training episode
            self.train_policy.update()

            # Add last loss to the result
            result.loss = self.last_loss
