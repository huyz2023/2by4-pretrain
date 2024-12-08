# Efficient 2:4 Sparse Pre-training
This repository provides the official implementations of efficient 2:4 pre-training toolkit from the following papers.

**Accelerating Transformer Pre-training with 2:4 Sparsity** [[arXiv]](https://arxiv.org/abs/2404.01847) [[OpenReview]](https://openreview.net/forum?id=kTaX87Zn6M) [[PDF]](https://proceedings.mlr.press/v235/hu24r.html)

Yuezhou Hu, Kang Zhao, Weiyu Huang, Jianfei Chen, Jun Zhu

International Conference on Machine Learning (ICML), 2024

**S-STE: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training** [[arXiv]](https://arxiv.org/abs/2409.09099) [[OpenReview]](https://openreview.net/forum?id=8abNCVJs2j)

Yuezhou Hu, Jun Zhu, Jianfei Chen

Neural Information Processing Systems (NeurIPS), 2024

For scripts to replicate the experimental results, please visit [https://github.com/thu-ml/2by4-pretrain-acc-examples](https://github.com/thu-ml/2by4-pretrain-acc-examples).

## Installation

From source:

```
git clone --recursive https://github.com/huyz2023/2by4-pretrain
cd 2by4-pretrain
pip install -e .
```

## Overview

To get started with 2:4-spMM, official [torch.sparse](https://pytorch.org/docs/2.1/sparse.html#sparse-semi-structured-tensors) works well enough. However, we've added more features on top of that.

**Constructing 2:4 tensor **

To construct a sparse semi-structured tensor, simply calling `sparse.to_sparse_semi_structured` would work:

```
import sparse

A = torch.randn(128, 128, device='cuda:0', dtype=torch.half)
A_sparse = sparse.to_sparse_semi_structured(A)
```

Different from PyTorch, this would automatically prune the smallest two elements out of four.

You can also specify a certain 2:4 mask for this step. Typically, the mask is a 0/1 tensor (dtype does not matter) which indicates how to prune the tensor:

```
A_sparse = sparse.to_sparse_semi_structured(A, mask=your_mask)
```

Additionally, our toolkit supports minimum-variance unbiased estimator (MVUE) as its pruning strategy:

```
A_sparse = sparse.to_sparse_semi_structured(A, MVUE24=True)
```

**Support for different dtype**

We now support float16, bfloat16, int8, float8_e5m2 and float8_e4m3fn in dense-sparse conversion. Let's try this out:

```
A = torch.randn(128, 128, device='cuda:0')
A_sparse = sparse.to_sparse_semi_structured(A, dtype=torch.float16)
A_sparse = sparse.to_sparse_semi_structured(A, dtype=torch.int8)
A_sparse = sparse.to_sparse_semi_structured(A, dtype=torch.float8_e5m2)
```

This will provide A_sparse in `dtype`, regardless of its original type.

**2:4 operations**

Same as PyTorch, those operations are supported:

- torch.addmm(bias, dense, sparse.t())
- torch.mm(dense, sparse)
- torch.mm(sparse, dense)
- aten.linear.default(dense, sparse, bias)
- aten.t.default(sparse)
- aten.t.detach(sparse)

There are two 2:4-spMM kernels in total. CUTLASS and cuSPARSElt. The cuSPARSElt backend is used only when you set

```
sparse.SparseSemiStructuredTensor._FORCE_CUTLASS = False
```

By default, this is always True.

**Transposable mask select**

Efficient mask select kernel based on convolution:

```
mask_select = sparse.TransposableSparse()
A_sparse, A_mask = sparse.mask_select(A)
```

**Masked decay**

Fused kernel for masked decay:

```
sparse.masked_add_(grad, p, mask, alpha=2e-4)
```

This is equivalent to `grad.add_(p.data * (1 - mask), alpha=decay)`.

**Soft-thresholding (pseudo)**

```
A_sparse, A_mask = sparse.soft_threshold24_triton(A)
```

### Accelerating Transformer Pre-training with 2:4 Sparsity

Take nanoGPT as an example.

**Step 1**

Replace `nn.Linear` with self-defined `SparseLinearTranspose`:

```
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
```

**Step 2**

Apply masked decay:

```
with torch.no_grad():
    for p in model.parameters():
        if hasattr(p, 'mask') and p.mode == 'sparse':
            p.grad = p.grad.float()
            masked_add_(p.grad.data, p.data, p.mask, alpha=alpha)
            p.cnt = 0
```

**Step 3**

Dense fine-tuning:

```
# Step 4: manually convert to dense fine-tune stage
if iter_num == 250000:
    for p in model.parameters():
        if hasattr(p, 'mask') and p.mode == 'sparse':
            p.mode = 'dense'
```

### S-STE: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training

Replace `nn.Linear` with self-defined `SparseLinear`:

```
class SoftThreshold(autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale):
        weight_temp = weight.detach()
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        return weight_sparse * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super(FP8SparseLinear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer('scale', torch.tensor(0.))

    def get_sparse_weights(self):
        return SoftThreshold.apply(self.weight, self.scale)

    @torch.no_grad()
    def init_scale(self):
        weight = self.weight.cuda()
        weight_temp = weight.detach()
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        weight.scale = torch.dot(torch.flatten(weight), torch.flatten(weight_sparse)) / torch.dot(
            torch.flatten(weight_sparse), torch.flatten(weight_sparse))
        self.scale.copy_(weight.scale.cpu())
        self.weight.scale = self.scale

    def forward(self, x):
        w = self.get_sparse_weights()
        x = F.linear(x, w, self.bias)
        return x
```

The relevant code of this can be found at [https://github.com/thu-ml/2by4-pretrain-acc-examples](https://github.com/thu-ml/2by4-pretrain-acc-examples).

## Citation

If you like our study, please cite:

```
@inproceedings{
  hu2024accelerating,
  title={Accelerating Transformer Pre-training with 2:4 Sparsity},
  author={Yuezhou Hu and Kang Zhao and Weiyu Huang and Jianfei Chen and Jun Zhu},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=kTaX87Zn6M}
}
@inproceedings{
  hu2024sste,
  title={S-{STE}: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training},
  author={Yuezhou Hu and Jun Zhu and Jianfei Chen},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=8abNCVJs2j}
}
```

