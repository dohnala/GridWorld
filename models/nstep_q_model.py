import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Model, ModelConfig


class NstepQModelConfig(ModelConfig):
    """
    N-step Q model's configuration.
    """

    def __init__(self, network, discount):
        """
        Initialize configuration.

        :param network: network
        :param discount: discount factor
        """
        super(NstepQModelConfig, self).__init__(network)

        self.discount = discount


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
        super(NstepQModel, self).__init__(input_shape, num_actions, config)

        self.steps = 0
        self.discount = config.discount
        self.target_sync = config.target_sync

        self.output = nn.Linear(self.network.output_shape(), self.num_actions)

        # Create target if sync is defined
        if target_sync:
            self.target = NstepQModel(input_shape, num_actions, None, config)
            self.__sync_target()

    def forward(self, states):
        result = Variable(torch.from_numpy(states))

        result = self.network(result)
        result = self.output(result)

        return result

    def update(self, states, actions, rewards, next_states, done):
        if done[-1][0]:
            # Final value = 0 if last next state is terminal
            final_value = 0
        else:
            # Use target model to estimate value
            target_model = self.target if self.target_sync else self

            # Final value is maximum q value for last next state
            final_value = target_model(next_states[-1]).detach().data.numpy().max()

        # Compute targets as discounted cumulative rewards
        targets = self.__discounted_cumulative_rewards__(rewards, final_value)

        # Compute model outputs for given states
        outputs = self.forward(states)

        # Turn actions into variable
        actions = Variable(torch.from_numpy(actions))

        # Compute predictions of actions in given states
        predictions = outputs.gather(1, actions).squeeze()

        # Turn targets into variable
        targets = Variable(torch.from_numpy(targets))

        # Compute Huber loss from predictions and targets
        loss = F.smooth_l1_loss(predictions, targets)

        # Perform optimization step to update parameter w.r.t given loss
        self.optimizer.step(loss)

        # Update steps
        self.steps += 1

        # Sync target
        if self.target_sync and self.steps % self.target_sync == 0:
            self.__sync_target()

        return loss.data[0]

    def state_dict(self, destination=None, prefix=''):
        # Filter out state of target
        return {k: v for k, v in super().state_dict(destination, prefix).items() if not k.startswith('target')}

    def named_parameters(self, memo=None, prefix=''):
        # Filter out target parameters
        return filter(lambda param: not param[0].startswith('target'), super().named_parameters(memo, prefix))

    def __discounted_cumulative_rewards__(self, rewards, final_value):
        targets = np.zeros(len(rewards))
        target = final_value

        # Compute targets from rewards
        for i in reversed(range(len(rewards))):
            target = rewards[i] + self.discount * target
            targets[i] = target

        return np.array(targets, dtype=np.float32)

    def __sync_target(self):
        self.target.load_state_dict(self.state_dict())
