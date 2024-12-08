""" AdamW Optimizer
Impl copied from PyTorch master
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class AdamW_STEP(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clipping=(None, None), option=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if option not in [1, 2]:
            raise ValueError("Invalid option: {}".format(option))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW_STEP, self).__init__(params, defaults)
        # TODO
        self.mask = 'dense'
        self.sliding_window = []
        self._variance_change = []
        self.T_omega = int(1 / (1 - betas[1]))
        self.eps = eps
        self.T_min, self.T_max = clipping
        self.t = 0
        self.option = option

    def __setstate__(self, state):
        super(AdamW_STEP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # TODO
        self._variance_change.clear()
        self.t += 1

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # part of v_t

                    # TODO
                    if hasattr(p, 'mask'):
                        assert p.mask == 'dense'
                        state['exp_avg_sq_last'] = torch.zeros_like(p.data)  # part of v_{t-1}

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                # TODO
                if hasattr(p, 'mask') and self.mask == 'dense':
                    assert p.mask == 'dense'
                    exp_avg_sq_last = state['exp_avg_sq_last']
                    exp_avg_sq_last.data.copy_(exp_avg_sq.data)

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # TODO
                if hasattr(p, 'mask') and self.mask == 'dense':
                    assert p.mask == 'dense'
                    exp_avg_sq_last = state['exp_avg_sq_last']
                    variance_change = exp_avg_sq - exp_avg_sq_last
                    self._variance_change.append(variance_change)

                # TODO
                if hasattr(p, 'mask') and self.mask == 'sparse':
                    assert p.mask == 'sparse'
                    preconditioned_variance = state['preconditioned_variance']
                    denom = (preconditioned_variance.sqrt()).add_(
                        math.sqrt(group['eps']) * math.sqrt(math.sqrt(group['eps'])))
                    # denom = torch.max(preconditioned_variance.sqrt(),
                    #                   torch.full_like(preconditioned_variance,
                    #                                   math.sqrt(group['eps']) * math.sqrt(math.sqrt(group['eps']))))
                    # denom = preconditioned_variance.add(group['eps']).sqrt_()

                    # step_size = group['lr'] / bias_correction1
                    #
                    # p.data.addcdiv_(-step_size, exp_avg, denom)
                    # p.data.add_((exp_avg / denom).nan_to_num_(0).clamp_(-2, 2), alpha=-step_size)

                else:
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                        # denom = (max_exp_avg_sq / bias_correction2).add_(group['eps']).sqrt_()
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                        # denom = (exp_avg_sq / bias_correction2).add_(group['eps']).sqrt_()

                    # step_size = group['lr'] / bias_correction1
                    #
                    # p.data.addcdiv_(-step_size, exp_avg, denom)

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        # TODO
        if self.mask == 'dense':
            # calculate Z_t and sliding window
            if self.option == 1:
                Z_t = torch.mean(torch.cat([e.view(-1) for e in self._variance_change]).abs())
            elif self.option == 2:
                Z_t = torch.exp(torch.mean(torch.log(torch.cat([e.view(-1) for e in self._variance_change]).abs())))
            self.sliding_window.append(Z_t)
            while len(self.sliding_window) > self.T_omega:
                self.sliding_window.pop(0)

            # check_for_switching_point
            average_Z = torch.mean(torch.stack(self.sliding_window))

            if (self.T_max is not None) and (self.T_min is not None):
                if (self.t > self.T_max) or (average_Z < self.eps and self.t > self.T_min):
                    self.mask = 'sparse'
            else:
                if average_Z < self.eps:
                    self.mask = 'sparse'

            if self.mask == 'sparse':
                print('Auto Switch Between two Phases...', flush=True)
                for group in self.param_groups:
                    for p in group['params']:
                        if hasattr(p, 'mask'):
                            assert p.mask == 'dense'
                            p.mask = 'sparse'
                            state = self.state[p]
                            state['preconditioned_variance'] = torch.empty_like(p.data)
                            preconditioned_variance = state['preconditioned_variance']
                            if amsgrad:
                                max_exp_avg_sq = state['max_exp_avg_sq']
                                preconditioned_variance.data.copy_(max_exp_avg_sq.data)
                            else:
                                exp_avg_sq = state['exp_avg_sq']
                                preconditioned_variance.data.copy_(exp_avg_sq.data)

        return loss
