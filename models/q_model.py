import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Model, ModelConfig
from networks import NetworkModule


class QModelConfig(ModelConfig):
    """
    Q model's configuration.
    """

    def __init__(self, base_network, discount):
        """
        Initialize configuration.

        :param base_network: base network
        :param discount: discount factor
        """
        super(QModelConfig, self).__init__(base_network)

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

        # Initialize output layer parameters
        self.__init_parameters__(self.output)

    def forward(self, states):
        result = Variable(torch.from_numpy(states), requires_grad=False)

        result = self.network(result)
        result = self.output(result)

        return result

    def output_shape(self):
        return self.num_actions

    @staticmethod
    def __init_parameters__(layer, std=1.0):
        weights = torch.randn(layer.weight.data.size())
        weights *= std / torch.sqrt(weights.pow(2).sum(1, keepdim=True))

        layer.weight.data = weights
        layer.bias.data.fill_(0)


class QModel(Model):
    """
    Q model used to update parameters using temporal difference error.
    """

    def __init__(self, input_shape, num_actions, target_sync, config):
        """
        Initialize model.

        :param input_shape: shape of input state
        :param num_actions: number of actions
        :param target_sync: after how many steps target network should be synced
        :param config: model's config
        """
        super(QModel, self).__init__(
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
        # Compute model outputs for given states
        outputs = self.predict(states)

        # Turn actions into variable
        actions = Variable(torch.from_numpy(actions), requires_grad=False)

        # Compute predictions of actions in given states
        predictions = outputs.gather(1, actions).squeeze()

        # Calculate targets
        targets = self.__calculate_targets__(rewards, next_states, done)

        # Compute Huber loss from predictions and targets
        loss = F.smooth_l1_loss(predictions, targets)

        # Perform optimization step to update parameter w.r.t given loss
        self.optimizer.step(loss, self.parameters())

        # Update steps
        self.steps += 1

        # Sync target network with main network
        if self.target_sync and self.steps % self.target_sync == 0:
            self.__sync_target_network__()

        return loss.data[0]

    def __calculate_targets__(self, rewards, next_states, done):
        """
        Calculate targets for model update.

        :param rewards: rewards
        :param next_states: next states
        :param done: done
        :return: targets
        """
        # Create masks to differentiate terminal states
        done_mask = torch.from_numpy(done)
        non_done_mask = 1 - done_mask

        # Turns rewards into variables
        rewards = Variable(torch.from_numpy(rewards))

        # Select only next states which are not terminal
        non_terminal_next_states = np.asarray([next_states[i] for i in range(len(done)) if not done[i][0]])

        # Create targets variable
        targets = Variable(torch.zeros(len(done), 1), requires_grad=False)

        # if there are any non terminal next states
        if non_terminal_next_states.size > 0:
            # Compute predictions using target network
            target_pred = self.__get_target_network__()(non_terminal_next_states)

            # Compute targets for non terminal states
            targets[non_done_mask] = rewards[non_done_mask] + self.discount * target_pred.max(1)[0]

        # Compute targets for terminal states
        targets[done_mask] = rewards[done_mask].detach()

        return targets

    def __get_target_network__(self):
        """
        Return target network to estimate targets.

        :return: target network
        """
        return self.target_network if self.target_sync else self.network

    def __sync_target_network__(self):
        """
        Synchronize target network with main network.

        :return: None
        """
        self.target_network.load_state_dict(self.network.state_dict())
