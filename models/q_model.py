import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Model


class QNstepModel(Model):
    """
    QModel using N-step algorithm to estimate targets.
    """

    def __init__(self, input_shape, num_actions, discount, hidden_units=None):
        super(QNstepModel, self).__init__(input_shape, num_actions)

        self.discount = discount
        self.hidden_units = hidden_units

        input_size = self.input_shape

        # Create hidden layers
        layers = []
        if self.hidden_units:
            for h in self.hidden_units:
                layers.append(nn.Linear(input_size, h))
                input_size = h

        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(input_size, self.num_actions)

    def forward(self, states):
        result = Variable(torch.from_numpy(states))

        for layer in self.hidden:
            result = F.relu(layer(result))

        result = self.output(result)

        return result

    def update(self, states, actions, rewards, next_states, done):
        if done[-1][0]:
            # Final value = 0 if last next state is terminal
            final_value = 0
        else:
            # Final value is maximum q value for last next state
            final_value = self.forward(next_states[-1]).data.numpy().max()

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

        return loss.data[0]

    def __discounted_cumulative_rewards__(self, rewards, final_value):
        targets = np.zeros(len(rewards))
        target = final_value

        # Compute targets from rewards
        for i in reversed(range(len(rewards))):
            target = rewards[i] + self.discount * target
            targets[i] = target

        return np.array(targets, dtype=np.float32)
