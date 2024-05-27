import torch

import triton
import triton.language as tl


@triton.jit
def masked_add_kernel(grad_ptr,
                      p_ptr,
                      p_mask_ptr,
                      n_elements,
                      alpha,
                      BLOCK_SIZE: tl.constexpr,
                      ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad += p * alpha
    tl.store(grad_ptr + offsets, grad, mask=mask)


def masked_add_(grad: torch.Tensor, p_data: torch.Tensor, p_mask: torch.Tensor, alpha: float = 0):
    '''
    equivalent to
    grad.add_(p.data * (1 - p.mask), alpha=decay)
    '''
    assert grad.is_cuda and p_data.is_cuda and p_mask.is_cuda
    assert (grad.layout, p_data.layout, p_mask.layout) == (torch.strided, torch.strided, torch.strided)
    assert grad.stride() == p_data.stride() == p_mask.stride()
    n_elements = grad.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    masked_add_kernel[grid](grad, p_data, p_mask, n_elements, alpha, BLOCK_SIZE=1024)


if __name__ == "__main__":
    grad = torch.tensor([1., 1., 1., 1.]).cuda()
    p = torch.tensor([1., 2., 3., 4.]).cuda()
    p_mask = torch.tensor([1., 0., 1., 0.]).cuda()
    alpha = 0.03
    masked_add_(grad, p, p_mask, alpha=0.03)
    print(grad)
