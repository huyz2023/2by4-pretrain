import torch
import triton
from torch import autograd
import triton.language as tl


@triton.jit
def tanh(x):
    tanh_neg = (tl.math.exp(x * 2) - 1) / (tl.math.exp(x * 2) + 1)
    tanh_pos = (1 - tl.math.exp(-2 * x)) / (1 + tl.math.exp(-2 * x))
    tanh = tl.where(x > 0, tanh_pos, tanh_neg)
    return tanh


@triton.jit
def _gelu_glu_fwd_kernel(
        output_ptr, input_ptr, output_row_stride, input_row_stride, output_col_stride, input_col_stride,
        output_page_stride, input_page_stride, n_pages, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    x = tl.load(input_ptr + row_idx * input_row_stride + col_idx * input_col_stride + tl.arange(0,
                                                                                                BLOCK_SIZE // 2) * input_page_stride,
                mask=tl.arange(0, BLOCK_SIZE // 2) < n_pages // 2, other=-float('inf'))
    gate = tl.load(input_ptr + row_idx * input_row_stride + col_idx * input_col_stride + (
            tl.arange(0, BLOCK_SIZE // 2) + n_pages // 2) * input_page_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 2) < n_pages // 2, other=-float('inf'))

    gate_cube = gate * gate * gate

    # beta = math.sqrt(2 / math.pi)
    beta = 0.7978845608028654
    kappa = 0.044715
    inner = beta * (gate + kappa * gate_cube)

    # # inner_tanh
    # inner_tanh_neg = (tl.math.exp(inner * 2) - 1) / (tl.math.exp(inner * 2) + 1)
    # inner_tanh_pos = (1 - tl.math.exp(-2 * inner)) / (1 + tl.math.exp(-2 * inner))
    # inner_tanh = tl.where(inner > 0, inner_tanh_pos, inner_tanh_neg)
    inner_tanh = tanh(inner)

    gate_gelu = 0.5 * gate * (inner_tanh + 1)
    gelu_glu = gate_gelu * x

    tl.store(output_ptr + row_idx * output_row_stride + col_idx * output_col_stride + tl.arange(0,
                                                                                                BLOCK_SIZE // 2) * output_page_stride,
             gelu_glu, mask=tl.arange(0, BLOCK_SIZE // 2) < n_pages // 2)


@triton.jit
def _gelu_glu_fwd_kernel_(
        output_ptr, input_ptr, output_row_stride, input_row_stride, output_col_stride, input_col_stride, n_rows, n_cols,
        BLOCK_SIZE: tl.constexpr
):
    col_idx = tl.program_id(0)
    row_idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + row_idx * input_row_stride + col_idx * input_col_stride,
                mask=tl.arange(0, BLOCK_SIZE) < n_rows, other=-float('inf'))
    gate = tl.load(input_ptr + row_idx * input_row_stride + (col_idx + n_cols // 2) * input_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE) < n_rows, other=-float('inf'))

    gate_cube = gate * gate * gate

    # beta = math.sqrt(2 / math.pi)
    beta = 0.7978845608028654
    kappa = 0.044715
    inner = beta * (gate + kappa * gate_cube)

    # # inner_tanh
    # inner_tanh_neg = (tl.math.exp(inner * 2) - 1) / (tl.math.exp(inner * 2) + 1)
    # inner_tanh_pos = (1 - tl.math.exp(-2 * inner)) / (1 + tl.math.exp(-2 * inner))
    # inner_tanh = tl.where(inner > 0, inner_tanh_pos, inner_tanh_neg)
    inner_tanh = tanh(inner)

    gate_gelu = 0.5 * gate * (inner_tanh + 1)
    gelu_glu = gate_gelu * x

    tl.store(output_ptr + row_idx * output_row_stride + col_idx * output_col_stride,
             gelu_glu, mask=tl.arange(0, BLOCK_SIZE) < n_rows)


@triton.jit
def _gelu_glu_bwd_kernel(grad_output_ptr, grad_input_ptr, input_ptr, grad_output_row_stride, grad_input_row_stride,
                         input_row_stride, grad_output_col_stride, grad_input_col_stride, input_col_stride,
                         grad_output_page_stride, grad_input_page_stride, input_page_stride, n_pages,
                         BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    grad_output = tl.load(
        grad_output_ptr + row_idx * grad_output_row_stride + col_idx * grad_output_col_stride + tl.arange(0,
                                                                                                          BLOCK_SIZE // 2) * grad_output_page_stride,
        mask=tl.arange(0, BLOCK_SIZE // 2) < n_pages // 2, other=-float('inf'))

    x = tl.load(input_ptr + row_idx * input_row_stride + col_idx * input_col_stride + tl.arange(0,
                                                                                                BLOCK_SIZE // 2) * input_page_stride,
                mask=tl.arange(0, BLOCK_SIZE // 2) < n_pages // 2, other=-float('inf'))
    gate = tl.load(input_ptr + row_idx * input_row_stride + col_idx * input_col_stride + (
            tl.arange(0, BLOCK_SIZE // 2) + n_pages // 2) * input_page_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 2) < n_pages // 2, other=-float('inf'))

    gate_cube = gate * gate * gate
    beta = 0.7978845608028654
    kappa = 0.044715
    inner = beta * (gate + kappa * gate_cube)
    inner_tanh = tanh(inner)
    gate_gelu = 0.5 * gate * (inner_tanh + 1)

    grad_x = grad_output * gate_gelu
    grad_gelu = grad_output * x

    grad_gate = grad_gelu * (0.5 * (1 + inner_tanh) + 0.5 * gate * (1 - inner_tanh * inner_tanh) * beta * (
            1 + kappa * 3 * gate * gate))

    tl.store(grad_input_ptr + row_idx * grad_input_row_stride + col_idx * grad_input_col_stride + tl.arange(0,
                                                                                                            BLOCK_SIZE // 2) * grad_input_page_stride,
             grad_x, mask=tl.arange(0, BLOCK_SIZE // 2) < n_pages // 2)
    tl.store(grad_input_ptr + row_idx * grad_input_row_stride + col_idx * grad_input_col_stride + (
            tl.arange(0, BLOCK_SIZE // 2) + n_pages // 2) * grad_input_page_stride,
             grad_gate, mask=tl.arange(0, BLOCK_SIZE // 2) < n_pages // 2)


@triton.jit
def _gelu_glu_bwd_kernel_(grad_output_ptr, grad_input_ptr, input_ptr, grad_output_row_stride, grad_input_row_stride,
                          input_row_stride, grad_output_col_stride, grad_input_col_stride, input_col_stride, n_rows,
                          n_cols,
                          BLOCK_SIZE: tl.constexpr):
    col_idx = tl.program_id(0)
    row_idx = tl.arange(0, BLOCK_SIZE)
    grad_output = tl.load(
        grad_output_ptr + row_idx * grad_output_row_stride + col_idx * grad_output_col_stride,
        mask=tl.arange(0, BLOCK_SIZE) < n_rows, other=-float('inf'))
    x = tl.load(input_ptr + row_idx * input_row_stride + col_idx * input_col_stride,
                mask=tl.arange(0, BLOCK_SIZE) < n_rows, other=-float('inf'))
    gate = tl.load(input_ptr + row_idx * input_row_stride + (col_idx + n_cols // 2) * input_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE) < n_rows, other=-float('inf'))

    gate_cube = gate * gate * gate
    beta = 0.7978845608028654
    kappa = 0.044715
    inner = beta * (gate + kappa * gate_cube)
    inner_tanh = tanh(inner)
    gate_gelu = 0.5 * gate * (inner_tanh + 1)

    grad_x = grad_output * gate_gelu
    grad_gelu = grad_output * x

    grad_gate = grad_gelu * (0.5 * (1 + inner_tanh) + 0.5 * gate * (1 - inner_tanh * inner_tanh) * beta * (
            1 + kappa * 3 * gate * gate))

    tl.store(grad_input_ptr + row_idx * grad_input_row_stride + col_idx * grad_input_col_stride,
             grad_x, mask=tl.arange(0, BLOCK_SIZE) < n_rows)
    tl.store(grad_input_ptr + row_idx * grad_input_row_stride + (col_idx + n_cols // 2) * grad_input_col_stride,
             grad_gate, mask=tl.arange(0, BLOCK_SIZE) < n_rows)


class gelu_glu(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        assert input.dim() == 3, 'input must be 3D'
        ctx.stride = input.stride()
        if ctx.stride[-1] == 1:
            n_rows, n_cols, n_pages = input.shape
            BLOCK_SIZE = triton.next_power_of_2(n_pages)
            num_warps = 4
            if BLOCK_SIZE >= 2048:
                num_warps = 8
            if BLOCK_SIZE >= 4096:
                num_warps = 16
            output = torch.empty(n_rows, n_cols, n_pages // 2, device=input.device, dtype=input.dtype)
            grid = (n_rows, n_cols, 1)
            _gelu_glu_fwd_kernel[grid](
                output, input, output.stride(0), input.stride(0), output.stride(1), input.stride(1), output.stride(2),
                input.stride(2), n_pages, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE,
            )
            ctx.save_for_backward(input)
            return output
        else:
            # fall back to 2D
            ctx.shape = input.shape
            input = input.view(-1, input.shape[-1])

            n_rows, n_cols = input.shape
            BLOCK_SIZE = triton.next_power_of_2(n_rows)
            num_warps = 4
            if BLOCK_SIZE >= 2048:
                num_warps = 8
            if BLOCK_SIZE >= 4096:
                num_warps = 16
            output = torch.empty(n_cols // 2, n_rows, device=input.device, dtype=input.dtype).t()
            _gelu_glu_fwd_kernel_[(n_cols // 2,)](
                output, input, output.stride(0), input.stride(0), output.stride(1), input.stride(1), n_rows, n_cols,
                num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE
            )
            ctx.save_for_backward(input)
            return output.view(*ctx.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.dim() == 3, 'grad_output must be 3D'
        if ctx.stride[-1] == 1:
            input = ctx.saved_tensors[0]
            grad_output = grad_output.contiguous()
            n_rows, n_cols, n_pages = grad_output.shape[0], grad_output.shape[1], grad_output.shape[2] * 2
            BLOCK_SIZE = triton.next_power_of_2(n_pages)
            num_warps = 4
            if BLOCK_SIZE >= 2048:
                num_warps = 8
            if BLOCK_SIZE >= 4096:
                num_warps = 16
            grad_input = torch.empty(n_rows, n_cols, n_pages, device=grad_output.device, dtype=grad_output.dtype)
            grid = (n_rows, n_cols, 1)
            _gelu_glu_bwd_kernel[grid](
                grad_output, grad_input, input, grad_output.stride(0), grad_input.stride(0), input.stride(0),
                grad_output.stride(1), grad_input.stride(1), input.stride(1), grad_output.stride(2),
                grad_input.stride(2), input.stride(2), n_pages, num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE,
            )
            return grad_input
        else:
            input = ctx.saved_tensors[0]
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            n_rows, n_cols = input.shape
            BLOCK_SIZE = triton.next_power_of_2(n_rows)
            num_warps = 4
            if BLOCK_SIZE >= 2048:
                num_warps = 8
            if BLOCK_SIZE >= 4096:
                num_warps = 16
            grad_input = torch.empty(n_cols, n_rows, device=input.device, dtype=input.dtype).t()
            _gelu_glu_bwd_kernel_[(n_cols // 2,)](
                grad_output, grad_input, input, grad_output.stride(0), grad_input.stride(0), input.stride(0),
                grad_output.stride(1), grad_input.stride(1), input.stride(1), n_rows, n_cols,
                num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE
            )
            return grad_input.view(ctx.shape)


class GELUglu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu_glu.apply(x)
