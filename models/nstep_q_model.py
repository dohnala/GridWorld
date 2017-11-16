import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Model, ModelConfig
from networks import NetworkModule


class NstepQModelConfig(ModelConfig):
    """
    N-step Q model's configuration.
    """

    def __init__(self, base_network, discount):
        """
        Initialize configuration.

        :param base_network: base network
        :param discount: discount factor
        """
        super(NstepQModelConfig, self).__init__(base_network)

        self.discount = discount


class QNetworkModule(NetworkModule):
    """
    Network module which outputs Q value for each action.
    """

    def __init__(self, input_shape, num_actions, base_network):
        """
        Initialize network module.

        :param input_shape: shape of input state
        :param num_actions: number of actions
        :param base_network: base network
        """
        super(QNetworkModule, self).__init__(input_shape)

        self.num_actions = num_actions

        self.network = base_network.build(input_shape)
        self.output = nn.Linear(self.network.output_shape(), self.num_actions)

    def forward(self, states):
        result = Variable(torch.from_numpy(states), requires_grad=False)

        result = self.network(result)
        result = self.output(result)

        return result

    def output_shape(self):
        return self.num_actions


class NstepQModel(Model):
    """
    Q model using N-step algorithm to estimate targets.
    """

    def __init__(self, input_shape, num_actions, target_sync, config):
        """
        Initialize agent.

        :param input_shape: shape of input state
        :param num_actions: number of actions
        :param target_sync: after how many steps target network should be synced
        :param config: model's config
        """
        super(NstepQModel, self).__init__(
            network=QNetworkModule(input_shape, num_actions, config.base_network),
            config=config)

        self.steps = 0
        self.discount = config.discount
        self.target_sync = config.target_sync

        # Create target network as copy of main network if sync is defined
        if target_sync:
            self.target_network = copy.deepcopy(self.network)

    def predict(self, states):
        """
        Predict Q values for all actions for given states.

        :param states: states
        :return: Q values of all actions
        """
        return self.network(states)

    def update(self, states, actions, rewards, next_states, done):
        """
        Update model using estimating target from given experience.

        :param states: states
        :param actions: actions
        :param rewards: rewards
        :param next_states: next states
        :param done: done flags
        :return: loss
        """
        if done[-1][0]:
            # Final value = 0 if last next state is terminal
            final_value = 0
        else:
            # Use target network to estimate value
            target_network = self.target_network if self.target_sync else self.network

            # Final value is maximum q value for last next state
            final_value = target_network(next_states[-1]).data.numpy().max()

        # Compute targets as discounted cumulative rewards
        targets = self.__discounted_cumulative_rewards__(rewards, final_value)

        # Compute model outputs for given states
        outputs = self.predict(states)

        # Turn actions into variable
        actions = Variable(torch.from_numpy(actions), requires_grad=False)

        # Compute predictions of actions in given states
        predictions = outputs.gather(1, actions).squeeze()

        # Turn targets into variable
        targets = Variable(torch.from_numpy(targets), requires_grad=False)

        # Compute Huber loss from predictions and targets
        loss = F.smooth_l1_loss(predictions, targets)

        # Perform optimization step to update parameter w.r.t given loss
        self.optimizer.step(loss)

        # Update steps
        self.steps += 1

        # Sync target network with main network
        if self.target_sync and self.steps % self.target_sync == 0:
            self.__sync_target_network__()

        return loss.data[0]

    def __discounted_cumulative_rewards__(self, rewards, final_value):
        targets = np.zeros(len(rewards))
        target = final_value

        # Compute targets from rewards
        for i in reversed(range(len(rewards))):
            target = rewards[i] + self.discount * target
            targets[i] = target

        return np.array(targets, dtype=np.float32)

    def __sync_target_network__(self):
        self.target_network.load_state_dict(self.network.state_dict())
