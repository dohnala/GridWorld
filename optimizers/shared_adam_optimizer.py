import math
import torch
import torch.optim as optim

from optimizers import Optimizer


class SharedAdam(optim.Adam):
    """
    Shared Adam implementation.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'][0]
                bias_correction2 = 1 - beta2 ** state['step'][0]
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class SharedAdamOptimizer(Optimizer):
    """
    Shared Adam optimizer.
    """

    def __init__(self, learning_rate):
        super(SharedAdamOptimizer, self).__init__()

        self.learning_rate = learning_rate
        self.shared_parameters = None
        self.optimizer = None

    def set_shared_parameters(self, parameters):
        self.shared_parameters = parameters
        self.optimizer = SharedAdam(parameters, lr=self.learning_rate)

    def step(self, loss, parameters):
        # Zero all gradients
        self.optimizer.zero_grad()

        # Compute all gradients w.r.t given loss
        loss.backward()

        # Ensure that given parameters share grads with share parameters
        self.__ensure_shared_grads__(parameters)

        # Update all variables with computed gradients
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict() if self.optimizer else {}

    def load_state_dict(self, state_dict):
        if self.optimizer:
            self.optimizer.load_state_dict(state_dict)

    def __ensure_shared_grads__(self, parameters):
        for param, shared_param in zip(parameters, self.shared_parameters):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad
