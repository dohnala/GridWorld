import numpy as np
import torch
from torch.autograd import Variable

from models import QModel


class NstepQModel(QModel):
    """
    Q model using N-step algorithm to estimate targets.
    """

    def __calculate_predictions__(self, states, actions):
        """
        Calculate model predictions.

        :param states: states
        :param actions: actions
        :return: predictions
        """
        # Calculate batch size as n_step * num_processes
        batch_size = int(np.prod(states.shape[:2]))
        state_shape = states.shape[2:]

        # Turn states into variable
        states = Variable(torch.from_numpy(states), requires_grad=False).view(batch_size, *state_shape)

        if self.use_cuda:
            states = states.cuda()

        # Compute model outputs for given states
        outputs = self.network(states)

        # Turn actions into variable
        actions = Variable(torch.from_numpy(actions), requires_grad=False).view(batch_size, -1)

        if self.use_cuda:
            actions = actions.cuda()

        # Compute predictions of actions in given states
        return outputs.gather(1, actions)

    def __calculate_targets__(self, rewards, next_states, dones):
        # Calculate batch size as n_step * num_processes
        batch_size = int(np.prod(rewards.shape[:2]))

        # Calculate last values
        last_values = self.__calculate_last_values(next_states, dones)

        # Compute targets as discounted cumulative rewards
        returns = self.__calculate_returns__(rewards, dones, last_values)

        # Turn targets into variable
        returns = Variable(torch.from_numpy(returns), requires_grad=False).view(batch_size, -1)

        if self.use_cuda:
            returns = returns.cuda()

        return returns

    def __calculate_last_values(self, next_states, dones):
        # Get last states
        last_states = Variable(torch.from_numpy(next_states[-1]), volatile=True)

        # Create mask from done flags of last states
        mask = 1 - dones[-1].astype(int)

        if self.use_cuda:
            last_states = last_states.cuda()

        # Predict maximum action values for last states using target network
        last_values = self.__get_target_network__()(last_states).data.cpu().numpy().max(axis=1, keepdims=True)

        # Multiply lat values by mask to zero values for terminal states
        last_values = last_values * mask

        return last_values

    def __calculate_returns__(self, rewards, dones, last_values):
        # Create returns with the same shape as rewards
        returns = np.zeros(rewards.shape, dtype=np.float32)

        r = last_values

        for i in reversed(range(len(rewards))):
            # Create mask from done flags for current step
            mask = 1 - dones[i].astype(int)

            # Compute returns for current step
            r = rewards[i] + self.discount * r * mask

            returns[i] = r

        return returns
