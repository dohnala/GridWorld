import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Model, ModelConfig
from networks import NetworkModule, Network


class QModelConfig(ModelConfig):
    """
    Q model's configuration.
    """

    def __init__(self, input_shape, num_actions, base_network, discount, target_sync=None, double_q=False,
                 use_cuda=False):
        """
        Initialize configuration.

        :param input_shape input shape
        :param num_actions number of actions
        :param base_network: base network
        :param discount: discount factor
        :param target_sync: after how many updates target network should be synced
        :param double_q: use double q learning
        :param use_cuda: use GPU
        """
        super(QModelConfig, self).__init__(input_shape, num_actions, base_network)

        assert type(discount) is float and 0.0 <= discount <= 1.0, "discount has to be float in [0, 1]"

        if target_sync is not None:
            assert type(target_sync) is int and target_sync > 0, "target_sync has to be integer greater than zero"

        if double_q:
            assert target_sync is not None, "target_sync has to be set in order to use double_q"

        assert type(use_cuda) is bool, "use_cuda has to be boolean"

        self.discount = discount
        self.target_sync = target_sync
        self.double_q = double_q
        self.use_cuda = use_cuda


class QNetwork(NetworkModule):
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
        super(QNetwork, self).__init__(input_shape)

        assert type(num_actions) is int and num_actions > 0, "num_actions is not valid"
        assert isinstance(base_network, Network), "network is not valid"

        self.num_actions = num_actions

        self.network = base_network.build(input_shape)
        self.output = nn.Linear(self.network.output_shape(), self.num_actions)

        # Initialize output layer parameters
        self.__init_parameters__(self.output)

    def forward(self, states):
        result = self.network(states)
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

    def __init__(self, config):
        """
        Initialize model.

        :param config: model's config
        """
        super(QModel, self).__init__(
            network=QNetwork(config.input_shape, config.num_actions, config.base_network),
            config=config)

        assert isinstance(config, QModelConfig), "config is not valid"

        self.updates = 0
        self.discount = config.discount
        self.target_sync = config.target_sync
        self.double_q = config.double_q
        self.use_cuda = config.use_cuda

        # Move network to GPU
        if self.use_cuda:
            self.network.cuda()

        # Create target network as copy of main network if sync is defined
        if self.target_sync:
            self.target_network = copy.deepcopy(self.network)

    def predict(self, states):
        """
        Predict Q values for all actions for given states.

        :param states: states
        :return: Q values of all actions
        """
        # Turn states into variable
        states = Variable(torch.from_numpy(states), volatile=True)

        # Move states variable into GPU
        if self.use_cuda:
            states = states.cuda()

        return self.network(states).data.cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones):
        """
        Update model using estimating target from given experience.

        :param states: states
        :param actions: actions
        :param rewards: rewards
        :param next_states: next states
        :param dones: done flags
        :return: loss
        """
        # Compute predictions
        predictions = self.__calculate_predictions__(states, actions)

        # Calculate targets
        targets = self.__calculate_targets__(rewards, next_states, dones)

        # Compute Huber loss from predictions and targets
        loss = F.smooth_l1_loss(predictions, targets)

        # Perform optimization step to update parameter w.r.t given loss
        self.optimizer.step(loss, self.parameters())

        # Increment number of updates
        self.updates += 1

        # Sync target network with main network
        if self.target_sync and self.updates % self.target_sync == 0:
            self.__sync_target_network__()

        return loss.data.cpu().numpy()[0]

    def __calculate_predictions__(self, states, actions):
        """
        Calculate model predictions.

        :param states: states
        :param actions: actions
        :return: predictions
        """
        # Turn states into variable
        states = Variable(torch.from_numpy(states), requires_grad=False)

        if self.use_cuda:
            states = states.cuda()

        # Compute model outputs for given states
        outputs = self.network(states)

        # Turn actions into variable
        actions = Variable(torch.from_numpy(actions), requires_grad=False)

        if self.use_cuda:
            actions = actions.cuda()

        # Compute predictions of actions in given states
        return outputs.gather(1, actions)

    def __calculate_targets__(self, rewards, next_states, dones):
        """
        Calculate targets for model update.

        :param rewards: rewards
        :param next_states: next states
        :param dones: done flags
        :return: targets
        """
        # Create masks to differentiate terminal states
        done_masks = torch.from_numpy(dones)

        if self.use_cuda:
            done_masks = done_masks.cuda()

        non_done_masks = 1 - done_masks

        # Turns rewards into variables
        rewards = Variable(torch.from_numpy(rewards))

        if self.use_cuda:
            rewards = rewards.cuda()

        # Select only next states which are not terminal
        non_terminal_next_states = np.asarray([next_states[i] for i in range(len(dones)) if not dones[i][0]])

        # Create targets variable
        targets = Variable(torch.zeros(len(dones), 1), requires_grad=False)

        if self.use_cuda:
            targets = targets.cuda()

        # if there are any non terminal next states
        if non_terminal_next_states.size > 0:
            # Turn non terminal next states into variable
            non_terminal_next_states = Variable(torch.from_numpy(non_terminal_next_states), volatile=True)

            if self.use_cuda:
                non_terminal_next_states = non_terminal_next_states.cuda()

            # Compute q values of next states using target network
            q_next = self.__get_target_network__()(non_terminal_next_states).detach()

            # If double q learning should be used
            if self.double_q:
                # Find indexes of best actions using main network
                _, best_indexes = self.network(non_terminal_next_states).detach().max(1)

                # Find values of best actions
                q_next = q_next.gather(1, best_indexes.unsqueeze(1)).squeeze(1)
            else:
                q_next, _ = q_next.max(1)

            # Switch volatile back to False, so the loss can be computed
            q_next.volatile = False

            # Compute targets for non terminal states
            targets[non_done_masks] = rewards[non_done_masks] + self.discount * q_next

        # Compute targets for terminal states
        targets[done_masks] = rewards[done_masks].detach()

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
