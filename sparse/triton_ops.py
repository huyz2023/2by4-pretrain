import random

import torch
import triton
import triton.language as tl

from .utils import _MVUE24_approx, _sparse24, _soft_threshold


@triton.jit
def _MVUE24_approx_triton(
        dense_ptr,
        sparse_ptr,
        dense_row_stride,
        sparse_row_stride,
        dense_col_stride,
        sparse_col_stride,
        m, k,
        seed,
        BLOCK_SIZE: tl.constexpr,
        ARRAY_LAYOUT: tl.constexpr
):
    if ARRAY_LAYOUT == 'row':
        row_idx = tl.program_id(0)
        col_idx = tl.program_id(1) * 4 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 4
        mask = col_idx < k
    elif ARRAY_LAYOUT == 'col':
        row_idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
        col_idx = tl.program_id(1) * 4
        mask = row_idx < m
    dense_40 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 0) * dense_col_stride, mask=mask)
    dense_41 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 1) * dense_col_stride, mask=mask)
    dense_42 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 2) * dense_col_stride, mask=mask)
    dense_43 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 3) * dense_col_stride, mask=mask)

    if ARRAY_LAYOUT == 'row':
        seed0 = seed + (tl.program_id(0) + tl.program_id(1) * m) * 2
        seed1 = seed + (tl.program_id(0) + tl.program_id(1) * m) * 2 + 1
    else:
        seed0 = seed + (tl.program_id(0) * k // 16 + tl.program_id(1)) * 2
        seed1 = seed + (tl.program_id(0) * k // 16 + tl.program_id(1)) * 2 + 1

    random0 = tl.rand(seed0, tl.arange(0, BLOCK_SIZE), n_rounds=5)
    random1 = tl.rand(seed1, tl.arange(0, BLOCK_SIZE), n_rounds=5)

    dense_40, dense_41, dense_42, dense_43, m0, m1, m2, m3 = _MVUE24_approx(dense_40, dense_41, dense_42, dense_43,
                                                                            random0, random1)

    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 0) * sparse_col_stride, dense_40, mask=mask & m0)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 1) * sparse_col_stride, dense_41, mask=mask & m1)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 2) * sparse_col_stride, dense_42, mask=mask & m2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 3) * sparse_col_stride, dense_43, mask=mask & m3)


def MVUE24_approx_triton(dense):
    m, k = dense.shape
    device = dense.device
    seed = random.randint(0, 2 ** 31 - 1)
    sparse = torch.zeros_like(dense)

    row_stride, col_stride = dense.stride()
    if row_stride > col_stride:
        array_layout = 'row'
        grid = lambda META: (m, triton.cdiv(k, 4 * META['BLOCK_SIZE']))
    else:
        array_layout = 'col'
        grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE']), k // 4,)
    func = _MVUE24_approx_triton
    func[grid](
        dense,
        sparse,
        dense.stride(0),
        sparse.stride(0),
        dense.stride(1),
        sparse.stride(1),
        m, k,
        seed,
        BLOCK_SIZE=1024,
        ARRAY_LAYOUT=array_layout
    )
    return sparse


@triton.jit
def _sparse24_triton(
        dense_ptr,
        sparse_ptr,
        mask_ptr,
        dense_row_stride,
        sparse_row_stride,
        mask_row_stride,
        dense_col_stride,
        sparse_col_stride,
        mask_col_stride,
        m, k,
        BLOCK_SIZE: tl.constexpr,
        ARRAY_LAYOUT: tl.constexpr
):
    if ARRAY_LAYOUT == 'row':
        row_idx = tl.program_id(0)
        col_idx = tl.program_id(1) * 4 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 4
        mask = col_idx < k
    elif ARRAY_LAYOUT == 'col':
        row_idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
        col_idx = tl.program_id(1) * 4
        mask = row_idx < m
    dense_40 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 0) * dense_col_stride, mask=mask)
    dense_41 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 1) * dense_col_stride, mask=mask)
    dense_42 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 2) * dense_col_stride, mask=mask)
    dense_43 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 3) * dense_col_stride, mask=mask)

    dense_40, dense_41, dense_42, dense_43, m0, m1, m2, m3 = _sparse24(dense_40, dense_41, dense_42, dense_43)

    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 0) * sparse_col_stride, dense_40, mask=mask & m0)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 1) * sparse_col_stride, dense_41, mask=mask & m1)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 2) * sparse_col_stride, dense_42, mask=mask & m2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 3) * sparse_col_stride, dense_43, mask=mask & m3)

    tl.store(mask_ptr + row_idx * mask_row_stride + (col_idx + 0) * mask_col_stride, m0, mask=mask & m0)
    tl.store(mask_ptr + row_idx * mask_row_stride + (col_idx + 1) * mask_col_stride, m1, mask=mask & m1)
    tl.store(mask_ptr + row_idx * mask_row_stride + (col_idx + 2) * mask_col_stride, m2, mask=mask & m2)
    tl.store(mask_ptr + row_idx * mask_row_stride + (col_idx + 3) * mask_col_stride, m3, mask=mask & m3)


def sparse24_triton(dense):
    m, k = dense.shape
    device = dense.device

    sparse = torch.zeros_like(dense)
    mask = torch.zeros_like(dense)

    row_stride, col_stride = dense.stride()
    if row_stride > col_stride:
        array_layout = 'row'
        grid = lambda META: (m, triton.cdiv(k, 4 * META['BLOCK_SIZE']))
    else:
        array_layout = 'col'
        grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE']), k // 4,)
    func = _sparse24_triton
    func[grid](
        dense,
        sparse,
        mask,
        dense.stride(0),
        sparse.stride(0),
        mask.stride(0),
        dense.stride(1),
        sparse.stride(1),
        mask.stride(1),
        m, k,
        BLOCK_SIZE=1024,
        ARRAY_LAYOUT=array_layout
    )
    return sparse, mask


@triton.jit
def _soft_threshold24_triton(
        dense_ptr,
        sparse_ptr,
        mask_ptr,
        dense_row_stride,
        sparse_row_stride,
        mask_row_stride,
        dense_col_stride,
        sparse_col_stride,
        mask_col_stride,
        m, k,
        BLOCK_SIZE: tl.constexpr,
        ARRAY_LAYOUT: tl.constexpr
):
    if ARRAY_LAYOUT == 'row':
        row_idx = tl.program_id(0)
        col_idx = tl.program_id(1) * 4 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 4
        mask = col_idx < k
    elif ARRAY_LAYOUT == 'col':
        row_idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
        col_idx = tl.program_id(1) * 4
        mask = row_idx < m
    dense_40 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 0) * dense_col_stride, mask=mask)
    dense_41 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 1) * dense_col_stride, mask=mask)
    dense_42 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 2) * dense_col_stride, mask=mask)
    dense_43 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 3) * dense_col_stride, mask=mask)

    dense_40, dense_41, dense_42, dense_43, m0, m1, m2, m3 = _soft_threshold(dense_40, dense_41, dense_42, dense_43)

    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 0) * sparse_col_stride, dense_40, mask=mask & m0)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 1) * sparse_col_stride, dense_41, mask=mask & m1)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 2) * sparse_col_stride, dense_42, mask=mask & m2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 3) * sparse_col_stride, dense_43, mask=mask & m3)

    tl.store(mask_ptr + row_idx * mask_row_stride + (col_idx + 0) * mask_col_stride, m0, mask=mask & m0)
    tl.store(mask_ptr + row_idx * mask_row_stride + (col_idx + 1) * mask_col_stride, m1, mask=mask & m1)
    tl.store(mask_ptr + row_idx * mask_row_stride + (col_idx + 2) * mask_col_stride, m2, mask=mask & m2)
    tl.store(mask_ptr + row_idx * mask_row_stride + (col_idx + 3) * mask_col_stride, m3, mask=mask & m3)


def soft_threshold24_triton(dense):
    m, k = dense.shape
    device = dense.device

    sparse = torch.zeros_like(dense)
    mask = torch.zeros_like(dense)

    row_stride, col_stride = dense.stride()
    if row_stride > col_stride:
        array_layout = 'row'
        grid = lambda META: (m, triton.cdiv(k, 4 * META['BLOCK_SIZE']))
    else:
        array_layout = 'col'
        grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE']), k // 4,)
    func = _soft_threshold24_triton
    func[grid](
        dense,
        sparse,
        mask,
        dense.stride(0),
        sparse.stride(0),
        mask.stride(0),
        dense.stride(1),
        sparse.stride(1),
        mask.stride(1),
        m, k,
        BLOCK_SIZE=1024,
        ARRAY_LAYOUT=array_layout
    )
    return sparse, mask
