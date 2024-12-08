import torch
from torch import autograd, nn
from torch.amp import custom_fwd, custom_bwd

from .triton_ops import MVUE24_approx_triton


class MyLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super(MyLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x):
        x = my_linear.apply(x, self.weight, self.bias)
        return x


class my_linear(autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = torch.mm(input, weight.t())
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        grad_output = grad_output
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = torch.mm(grad_output, weight).view(ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = torch.mm(grad_output.t(), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x):
        x = sparse_linear.apply(x, self.weight, self.bias)
        return x


class sparse_linear(autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = torch.mm(input, weight.t())
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        grad_output = grad_output
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = torch.mm(grad_output, MVUE24_approx_triton(weight.t()).t()).view(ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = torch.mm(grad_output.t(), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias
