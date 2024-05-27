import torch
from torch import autograd, nn, autocast
import torch.nn.functional as F

from torch.cuda.amp import custom_fwd, custom_bwd

from sparse.semi_structured import to_sparse_semi_structured, SparseSemiStructuredTensor
from sparse.transposable_semi_structured import TransposableSparse


class SparseLinearTranspose(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, func=lambda step: 'dense',
                 **kwargs):
        super(SparseLinearTranspose, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.weight.freq = 40  # update freq

        self.weight.cnt = 0  # how many steps after an optim step
        self.weight.counter = 0  # how many optim steps
        self.weight.step = 0  # total training step

        self.weight.mask = torch.ones_like(self.weight, dtype=torch.bool)
        self.weight.weight_sparse = None
        self.weight.weight_sparse_T = None
        self.weight.mode = 'sparse'
        self.func = func

        self.transposable_sparse = TransposableSparse(abs=True)
        SparseSemiStructuredTensor._FORCE_CUTLASS = True  # we won't need this later

    def forward(self, x):
        if self.weight.mode == 'dense':
            x = F.linear(x, self.weight, self.bias)
        else:
            self.weight.mask = self.weight.mask.to(device=self.weight.device)
            if self.weight.counter % self.weight.freq == 0 and self.weight.cnt == 0:
                _, self.weight.mask = self.transposable_sparse(self.weight)
            if self.weight.cnt == 0:
                self.weight.weight_sparse = to_sparse_semi_structured(self.weight, mask=self.weight.mask,
                                                                      dtype=torch.float16)
                self.weight.weight_sparse_T = to_sparse_semi_structured(self.weight.T, mask=self.weight.mask.T,
                                                                        dtype=torch.float16)
            with autocast(device_type='cuda', dtype=torch.float16):
                x = sparse_linear_transpose.apply(x, self.weight, self.weight.weight_sparse,
                                                  self.weight.weight_sparse_T,
                                                  self.bias)

        if self.training:
            if self.weight.cnt == 0:
                self.weight.counter += 1
            self.weight.step += 1
            self.weight.cnt += 1
        return x


class sparse_linear_transpose(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, weight_sparse, weight_sparse_T, bias):
        ctx.save_for_backward(input, weight_sparse_T, bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = torch.mm(input, weight_sparse.t())
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output
        input, weight_T, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            if grad_output.stride() == (0, 0, 0):
                grad_output = torch.ones_like(grad_output, device=grad_output.device, dtype=grad_output.dtype)
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_input = torch.mm(grad_output, weight_T.t()).view(
                ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = torch.mm(to_sparse_semi_structured(grad_output.t(), MVUE24=True), input)
        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, None, None, grad_bias
