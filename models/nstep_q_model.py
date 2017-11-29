import numpy as np
import torch
from torch.autograd import Variable

from models import QModel


class NstepQModel(QModel):
    """
    Q model using N-step algorithm to estimate targets.
    """

    def __calculate_targets__(self, rewards, next_states, done):
        if done[-1][0]:
            # Final value = 0 if last next state is terminal
            final_value = 0
        else:
            # Use target network to estimate value
            # Final value is maximum q value for last next state
            last_state = Variable(torch.from_numpy(np.expand_dims(next_states[-1], axis=0)), requires_grad=False)

            if self.use_cuda:
                last_state = last_state.cuda()

            final_value = self.__get_target_network__()(last_state).data.cpu().numpy().max()

        # Compute targets as discounted cumulative rewards
        targets = self.__discounted_cumulative_rewards__(rewards, final_value)

        # Turn targets into variable
        targets = Variable(torch.from_numpy(targets), requires_grad=False)

        if self.use_cuda:
            targets = targets.cuda()

        return targets

    def __discounted_cumulative_rewards__(self, rewards, final_value):
        targets = np.zeros(len(rewards))
        target = final_value

        # Compute targets from rewards
        for i in reversed(range(len(rewards))):
            target = rewards[i] + self.discount * target
            targets[i] = target

        return np.array(targets, dtype=np.float32)
