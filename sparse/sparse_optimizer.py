import torch

import triton
import triton.language as tl


def get_configs():
    configs = []
    for block in [32, 64, 128]:
        for num_stages in [2, 3, 4, 5]:
            for num_warps in [2, 4, 8]:
                configs.append(triton.Config({'BLOCK_SIZE': block}, num_stages=num_stages, num_warps=num_warps))

    return configs


@triton.autotune(
    configs=get_configs(),
    key=['m'],
)
@triton.jit
def _inverse(
        F_ptr,
        out_ptr,
        F_row_stride,
        out_row_stride,
        F_col_stride,
        out_col_stride,
        F_page_stride,
        out_page_stride,
        m,  # F.shape
        BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < m

    a11 = tl.load(F_ptr + row_idx * F_row_stride + 0 * F_col_stride + 0 * F_page_stride, mask=mask)
    a12 = tl.load(F_ptr + row_idx * F_row_stride + 0 * F_col_stride + 1 * F_page_stride, mask=mask)
    a13 = tl.load(F_ptr + row_idx * F_row_stride + 0 * F_col_stride + 2 * F_page_stride, mask=mask)
    a14 = tl.load(F_ptr + row_idx * F_row_stride + 0 * F_col_stride + 3 * F_page_stride, mask=mask)
    a21 = tl.load(F_ptr + row_idx * F_row_stride + 1 * F_col_stride + 0 * F_page_stride, mask=mask)
    a22 = tl.load(F_ptr + row_idx * F_row_stride + 1 * F_col_stride + 1 * F_page_stride, mask=mask)
    a23 = tl.load(F_ptr + row_idx * F_row_stride + 1 * F_col_stride + 2 * F_page_stride, mask=mask)
    a24 = tl.load(F_ptr + row_idx * F_row_stride + 1 * F_col_stride + 3 * F_page_stride, mask=mask)
    a31 = tl.load(F_ptr + row_idx * F_row_stride + 2 * F_col_stride + 0 * F_page_stride, mask=mask)
    a32 = tl.load(F_ptr + row_idx * F_row_stride + 2 * F_col_stride + 1 * F_page_stride, mask=mask)
    a33 = tl.load(F_ptr + row_idx * F_row_stride + 2 * F_col_stride + 2 * F_page_stride, mask=mask)
    a34 = tl.load(F_ptr + row_idx * F_row_stride + 2 * F_col_stride + 3 * F_page_stride, mask=mask)
    a41 = tl.load(F_ptr + row_idx * F_row_stride + 3 * F_col_stride + 0 * F_page_stride, mask=mask)
    a42 = tl.load(F_ptr + row_idx * F_row_stride + 3 * F_col_stride + 1 * F_page_stride, mask=mask)
    a43 = tl.load(F_ptr + row_idx * F_row_stride + 3 * F_col_stride + 2 * F_page_stride, mask=mask)
    a44 = tl.load(F_ptr + row_idx * F_row_stride + 3 * F_col_stride + 3 * F_page_stride, mask=mask)

    det = a11 * a22 * a33 * a44 - \
          a12 * a23 * a34 * a41 + \
          a13 * a24 * a31 * a42 - \
          a14 * a21 * a32 * a43 + \
          a14 * a23 * a32 * a41 - \
          a11 * a24 * a33 * a42 + \
          a12 * a21 * a34 * a43 - \
          a13 * a22 * a31 * a44 + \
 \
          a12 * a23 * a31 * a44 - \
          a13 * a21 * a34 * a42 + \
          a11 * a24 * a32 * a43 - \
          a14 * a22 * a33 * a41 + \
          a14 * a21 * a33 * a42 - \
          a12 * a24 * a31 * a43 + \
          a13 * a22 * a34 * a41 - \
          a11 * a23 * a32 * a44 + \
 \
          a13 * a21 * a32 * a44 - \
          a11 * a22 * a34 * a43 + \
          a12 * a24 * a33 * a41 - \
          a14 * a23 * a31 * a42 + \
          a14 * a22 * a31 * a43 - \
          a13 * a24 * a32 * a41 + \
          a11 * a23 * a34 * a42 - \
          a12 * a21 * a33 * a44

    # a11 a12 a13 a14
    # a21 a22 a23 a24
    # a31 a32 a33 a34
    # a41 a42 a43 a44

    # Adjugate matrix
    c11 = (a22 * a33 * a44
           + a23 * a34 * a42
           + a24 * a32 * a43
           - a24 * a33 * a42
           - a23 * a32 * a44
           - a22 * a34 * a43)
    c12 = -(a21 * a33 * a44
            + a23 * a34 * a41
            + a24 * a31 * a43
            - a24 * a33 * a41
            - a23 * a31 * a44
            - a21 * a34 * a43)
    c13 = (a21 * a32 * a44
           + a22 * a34 * a41
           + a24 * a31 * a42
           - a24 * a32 * a41
           - a22 * a31 * a44
           - a21 * a34 * a42)
    c14 = -(a21 * a32 * a43
            + a22 * a33 * a41
            + a23 * a31 * a42
            - a23 * a32 * a41
            - a22 * a31 * a43
            - a21 * a33 * a42)
    c21 = -(a12 * a33 * a44
            + a13 * a34 * a42
            + a14 * a32 * a43
            - a14 * a33 * a42
            - a13 * a32 * a44
            - a12 * a34 * a43)
    c22 = (a11 * a33 * a44
           + a13 * a34 * a41
           + a14 * a31 * a43
           - a14 * a33 * a41
           - a13 * a31 * a44
           - a11 * a34 * a43)
    c23 = -(a11 * a32 * a44
            + a12 * a34 * a41
            + a14 * a31 * a42
            - a14 * a32 * a41
            - a12 * a31 * a44
            - a11 * a34 * a42)
    c24 = (a11 * a32 * a43
           + a12 * a33 * a41
           + a13 * a31 * a42
           - a13 * a32 * a41
           - a12 * a31 * a43
           - a11 * a33 * a42)
    c31 = (a12 * a23 * a44
           + a13 * a24 * a42
           + a14 * a22 * a43
           - a14 * a23 * a42
           - a13 * a22 * a44
           - a12 * a24 * a43)
    c32 = -(a11 * a23 * a44
            + a13 * a24 * a41
            + a14 * a21 * a43
            - a14 * a23 * a41
            - a13 * a21 * a44
            - a11 * a24 * a43)
    c33 = (a11 * a22 * a44
           + a12 * a24 * a41
           + a14 * a21 * a42
           - a14 * a22 * a41
           - a12 * a21 * a44
           - a11 * a24 * a42)
    c34 = -(a11 * a22 * a43
            + a12 * a23 * a41
            + a13 * a21 * a42
            - a13 * a22 * a41
            - a12 * a21 * a43
            - a11 * a23 * a42)
    c41 = -(a12 * a23 * a34
            + a13 * a24 * a32
            + a14 * a22 * a33
            - a14 * a23 * a32
            - a13 * a22 * a34
            - a12 * a24 * a33)
    c42 = (a11 * a23 * a34
           + a13 * a24 * a31
           + a14 * a21 * a33
           - a14 * a23 * a31
           - a13 * a21 * a34
           - a11 * a24 * a33)
    c43 = -(a11 * a22 * a34
            + a12 * a24 * a31
            + a14 * a21 * a32
            - a14 * a22 * a31
            - a12 * a21 * a34
            - a11 * a24 * a32)
    c44 = (a11 * a22 * a33
           + a12 * a23 * a31
           + a13 * a21 * a32
           - a13 * a22 * a31
           - a12 * a21 * a33
           - a11 * a23 * a32)

    tl.store(out_ptr + row_idx * out_row_stride + 0 * out_col_stride + 0 * out_page_stride, c11 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 0 * out_col_stride + 1 * out_page_stride, c21 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 0 * out_col_stride + 2 * out_page_stride, c31 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 0 * out_col_stride + 3 * out_page_stride, c41 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 1 * out_col_stride + 0 * out_page_stride, c12 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 1 * out_col_stride + 1 * out_page_stride, c22 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 1 * out_col_stride + 2 * out_page_stride, c32 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 1 * out_col_stride + 3 * out_page_stride, c42 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 2 * out_col_stride + 0 * out_page_stride, c13 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 2 * out_col_stride + 1 * out_page_stride, c23 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 2 * out_col_stride + 2 * out_page_stride, c33 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 2 * out_col_stride + 3 * out_page_stride, c43 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 3 * out_col_stride + 0 * out_page_stride, c14 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 3 * out_col_stride + 1 * out_page_stride, c24 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 3 * out_col_stride + 2 * out_page_stride, c34 / det, mask=mask)
    tl.store(out_ptr + row_idx * out_row_stride + 3 * out_col_stride + 3 * out_page_stride, c44 / det, mask=mask)


def inverse(Fisher):
    m, _, _ = Fisher.shape
    device = Fisher.device
    out = torch.empty_like(Fisher)
    grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE']),)
    _inverse[grid](
        Fisher,
        out,
        Fisher.stride(0),
        out.stride(0),
        Fisher.stride(1),
        out.stride(1),
        Fisher.stride(2),
        out.stride(2),
        m
    )
    return out


""" Adam Optimizer
Impl copied from PyTorch master
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
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

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
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
                    state['exp_avg_sq'] = torch.zeros_like(p.data) if getattr(p, 'cnt', None) is None \
                        else torch.zeros(p.data.numel() // 4, 4, 4, device=p.data.device, dtype=p.data.dtype)  #
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) if getattr(p, 'cnt', None) is None else \
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, torch.matmul(grad.view(-1, 4, 1), grad.view(-1, 1, 4)))  #
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps']) \
                        if getattr(p, 'cnt', None) is None else (exp_avg_sq / bias_correction2).add_(
                        torch.eye(4, device=exp_avg_sq.device).unsqueeze_(0).mul_(group['eps']))  #

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom) if getattr(p, 'cnt', None) is None else p.data.add_(
                    -step_size, torch.matmul(torch.inverse(denom), exp_avg.view(-1, 4, 1)).view(p.data.shape))  #

        return loss


class AdamW(Optimizer):
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
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
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
                    state['exp_avg_sq'] = torch.zeros_like(p.data) if getattr(p, 'cnt', None) is None \
                        else torch.zeros(p.data.numel() // 4, 4, 4, device=p.data.device, dtype=p.data.dtype)  #
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) if getattr(p, 'cnt', None) is None else \
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, torch.matmul(grad.view(-1, 4, 1), grad.view(-1, 1, 4)))  #
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps']) \
                        if getattr(p, 'cnt', None) is None else (exp_avg_sq / bias_correction2).add_(
                        torch.eye(4, device=exp_avg_sq.device).unsqueeze_(0).mul_(group['eps']))  #

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom) if getattr(p, 'cnt', None) is None else p.data.add_(
                    -step_size, torch.matmul(torch.inverse(denom), exp_avg.view(-1, 4, 1)).view(p.data.shape))  #

        return loss
