import random
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


def _sparse_semi_structured_from_dense(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if m % 32 != 0:
        raise RuntimeError(
            f"Number rows columns of dense matrix {m} must be divisible by 32"
        )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )
    meta_ncols = k // (4 * quadbits_per_meta_elem)

    dense_4 = dense.view(-1, k // 4, 4)
    # m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    A, B, C, D = dense_4.abs().unbind(-1)
    x1, x2, x3, x4, x5, x6 = A > B, A > C, A > D, B > C, B > D, C > D
    m0, m1, m2, m3 = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1101
    #     [False, True,  True,  True ] -> 0b1001
    #     [True,  False, False, False] -> 0b1100
    #     [True,  False, True,  True ] -> 0b1000
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b1000
    #     [True,  True,  True,  True ] -> 0b1000
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.

    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = bit1 | ~m2
    bit3 = bit0 | ~m1 | m2
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))
    sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
    sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = (
                meta_n[:, :, 0]
                | (meta_n[:, :, 1] << 4)
                | (meta_n[:, :, 2] << 8)
                | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
                meta_n[:, :, 0]
                | (meta_n[:, :, 1] << 4)
                | (meta_n[:, :, 2] << 8)
                | (meta_n[:, :, 3] << 12)
                | (meta_n[:, :, 4] << 16)
                | (meta_n[:, :, 5] << 20)
                | (meta_n[:, :, 6] << 24)
                | (meta_n[:, :, 7] << 28)
        )

    # Metadata values are now to be reshuffled in a way given in
    # reorder_meta() function, in
    # tools/util/include/cutlass/util/host_reorder.h file of CUTLASS
    # source tree.  Furthermore, CUTLASS template for sparse GEMM
    # decides upon layout of this matrix, and at the moment for the
    # sparse GEMM executed on tensor cores, this is layout described
    # by ColumnMajorInterleaved<2> data structure, in
    # include/cutlass/layout/matrix.h of CUTLASS source tree.  The
    # reordering of meta matrix into meta_reordered matrix calculated
    # according to these segments of CUTLASS code is given below.
    # However, this calculation produces offsets for scatter access
    # from metadata matrix to redordered metadata matrix, and gather
    # pattern is more efficient.  For this reason, the scatter offsets
    # are reverted and printed, through enabling commented block at
    # the end of following code.  Resulting gather offsets are then
    # analyzed, on several (m, k) value pairs (in particular: (32,
    # 128), (32, 256), (64, 128) and (64, 256)), and the code that
    # follows this comment is written to reproduce these gather offsets.
    #
    #    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    #    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)
    #
    #    # Reorder the rows, then swizzle the 2x2 blocks.
    #    group = 32 if meta_dtype.itemsize == 2 else 16
    #    interweave = 4 if meta_dtype.itemsize == 2 else 2
    #    dst_rows = (
    #        dst_rows // group * group
    #        + (dst_rows % 8) * interweave
    #        + (dst_rows % group) // 8
    #    )
    #
    #    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    #    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    #    dst_rows += topright - bottomleft
    #    dst_cols -= topright - bottomleft
    #
    #    # Assumed that meta tensor is to be stored in CUTLASS
    #    # InterleavedColumnMajor layout, and reverse engineered
    #    # corresponding code to store values into this tensor.
    #    interleave = 2
    #    cols_maj = dst_cols // interleave
    #    cols_min = dst_cols % interleave
    #    meta_reordered_offsets = (
    #        cols_maj * m * interleave + dst_rows * interleave + cols_min
    #    )
    #
    #    meta_reordered = torch.empty((m, meta_ncols), dtype=meta_dtype, device=device)
    #    meta_reordered.view(-1)[meta_reordered_offsets.view(-1)] = meta.view(-1)
    #
    #    # Uncomment to have gather pattern for meta_reordered printed
    #    #
    #    #offsets = torch.empty(
    #    #    (m, meta_ncols), dtype=meta_reordered_offsets.dtype, device=device
    #    #)
    #    #offsets.view(-1)[meta_reordered_offsets.view(-1)] = torch.arange(
    #    #    0, m * meta_ncols, dtype=meta_reordered_offsets.dtype, device=device
    #    #)
    #    #torch.set_printoptions(threshold=1000000)
    #    #print("------------------------------------------------------------")
    #    #print("dtype =", dtype, ", m =", m, ", k =", k, ", meta_ncols =", meta_ncols)
    #    #print(offsets.view(-1))
    #

    # No point to try to understand this code: as mentioned in the
    # comment above it is written to reproduce gather offsets, as
    # these would be calculated by CUTLASS, and to be efficient, but
    # it contains several magic values and magic calculations that
    # make it rather hard to read, let alone understand.
    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = 32
        magic2 = 16
        magic3 = k // 2
        magic4 = [0, k // 4, 1, k // 4 + 1]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = 64
        magic2 = 32
        magic3 = 2 * k
        magic4 = [0, k // 2, 1, k // 2 + 1, k, 3 * k // 2, k + 1, 3 * k // 2 + 1]
    tmp0 = torch.zeros(m * meta_ncols, dtype=torch.int64, device=device)
    tmp1 = (
            tmp0.view(meta_ncols // 2, -1)
            + torch.arange(0, meta_ncols, 2, device=device).view(meta_ncols // 2, 1)
    ).view(-1, magic1)
    tmp2 = (
        (
                torch.arange(0, 8, device=device).view(-1, 1)
                * torch.ones((magic0,), dtype=torch.int64, device=device)
                * meta_ncols
        )
        .view(-1)
        .repeat(m * meta_ncols // magic1)
        .view(-1, magic1)
    )
    tmp3 = (torch.arange(0, m // magic2, device=device).view(-1, 1) * magic3).repeat(
        meta_ncols // 2, magic1
    )
    tmp4 = torch.tensor(magic4, device=device).repeat(tmp3.shape[0], 8)
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4

    meta_reordered = torch.gather(meta.view(-1), 0, meta_offsets.view(-1)).view(
        m, meta_ncols
    )
    return (sparse, meta_reordered)


def _sparse_semi_structured_to_dense(sparse, meta_reordered):
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    m, k = sparse.shape
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4

    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}"
        )
    if meta_ncols * 4 * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * 4 * quadbits_per_meta_elem // 2}, "
            "expected according to the number of columns of meta matrix"
        )

    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = [0, 1, 32, 33]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = [0, 1, 4, 5]
    tmp1 = torch.tensor([0, 2], dtype=torch.int64, device=device).repeat(
        meta_nrows, meta_ncols // 2
    )
    tmp2 = (
        (torch.arange(0, meta_ncols // 2, device=device) * 2 * meta_nrows)
        .view(-1, 1)
        .repeat(1, 2)
        .view(-1)
        .repeat(m, 1)
    )
    tmp3 = (
        (torch.arange(0, 8, device=device) * magic0)
        .view(-1, 1)
        .repeat(m // 8, meta_ncols)
    )
    tmp4 = (
        torch.tensor(magic1, device=device)
        .view(-1, 1)
        .repeat(1, 8 * meta_ncols)
        .repeat(meta_nrows // 32, 1)
        .view(meta_nrows, meta_ncols)
    )
    tmp5 = (
        (torch.arange(0, meta_nrows // 32, device=device) * 64)
        .view(-1, 1)
        .repeat(1, 32 * meta_ncols)
        .view(meta_nrows, meta_ncols)
    )
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4 + tmp5

    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets.view(-1)).view(
        m, meta_ncols
    )

    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem), dtype=meta_dtype, device=device
    )
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
        meta_2[:, :, 8] = (meta >> 16) & 0b11
        meta_2[:, :, 9] = (meta >> 18) & 0b11
        meta_2[:, :, 10] = (meta >> 20) & 0b11
        meta_2[:, :, 11] = (meta >> 22) & 0b11
        meta_2[:, :, 12] = (meta >> 24) & 0b11
        meta_2[:, :, 13] = (meta >> 26) & 0b11
        meta_2[:, :, 14] = (meta >> 28) & 0b11
        meta_2[:, :, 15] = (meta >> 30) & 0b11

    dense_offsets = meta_2.view(-1) + (
            torch.arange(0, m * k // 2, device=device) * 4
    ).view(-1, 1).repeat(1, 2).view(-1)

    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    dense.scatter_(0, dense_offsets, sparse.view(-1))

    return dense.view(m, 2 * k)


def sparse_semi_structured_from_dense(dense):
    from torch._dynamo.utils import is_compile_supported
    if is_compile_supported(dense.device.type):
        kernel = torch.compile(_sparse_semi_structured_from_dense)
        return kernel(dense)

    return _sparse_semi_structured_from_dense(dense)


def sparse_semi_structured_to_dense(sparse, meta_reordered):
    from torch._dynamo.utils import is_compile_supported
    if is_compile_supported(sparse.device.type):
        kernel = torch.compile(_sparse_semi_structured_to_dense)
        return kernel(sparse, meta_reordered)

    return _sparse_semi_structured_to_dense(sparse, meta_reordered)


def get_configs():
    configs = []
    for block in [8, 16]:
        for num_stages in [1, 2, 3, 4, 5, 6, 7]:
            for num_warps in [1, 2, 4, 8, 16]:
                configs.append(triton.Config({'BLOCK_SIZE': block}, num_stages=num_stages, num_warps=num_warps))

    for block in [32, 64, 128, 256]:
        for num_stages in [1, 2, 3, 4, 5, 6, 7]:
            for num_warps in [1, 2, 4, 8, 16]:
                configs.append(triton.Config({'BLOCK_SIZE': block}, num_stages=num_stages, num_warps=num_warps))
    return configs


# def get_cast_configs():
#     configs = []
#     for block in [1024, 2048, 4096]:
#         for num_stages in [1, 2, 3, 4, 5, 6, 7]:
#             for num_warps in [1, 2, 4, 8, 16]:
#                 if num_warps <= block // 128:
#                     configs.append(triton.Config({'BLOCK_SIZE': block}, num_stages=num_stages, num_warps=num_warps))
#     return configs


@triton.jit
def _MVUE24_approx(x0, x1, x2, x3,
                   random0, random1):
    eps = 1.19209e-07
    a0 = tl.abs(x0) + eps
    a1 = tl.abs(x1) + eps
    a2 = tl.abs(x2) + eps
    a3 = tl.abs(x3) + eps
    sum = a0 + a1 + a2 + a3

    t0 = a0 / sum
    t1 = a1 / sum
    t2 = a2 / sum
    t3 = a3 / sum

    s0 = sum - a0
    s1 = sum - a1
    s2 = sum - a2
    s3 = sum - a3

    k0 = t0 / s0
    k1 = t1 / s1
    k2 = t2 / s2
    k3 = t3 / s3
    k = k0 + k1 + k2 + k3

    p0 = (t0 + a0 * (k - k0))
    p1 = (t1 + a1 * (k - k1))
    p2 = (t2 + a2 * (k - k2))
    p3 = (t3 + a3 * (k - k3))

    m0 = (random0 <= t0)
    m1 = ((random0 <= (t0 + t1)) & ~m0)
    m2 = ((random0 <= (t0 + t1 + t2)) & ~m1 & ~m0)
    m3 = ~m2 & ~m1 & ~m0

    d_a0 = ~m0 * a0
    d_a1 = ~m1 * a1
    d_a2 = ~m2 * a2
    d_a3 = ~m3 * a3
    d_sum = d_a0 + d_a1 + d_a2 + d_a3

    t = random1 * d_sum
    d_m0 = (t <= d_a0)
    d_m1 = ((t <= (d_a0 + d_a1)) & ~d_m0)
    d_m2 = ((t <= (d_a0 + d_a1 + d_a2)) & ~d_m1 & ~d_m0)
    d_m3 = ~d_m2 & ~d_m1 & ~d_m0

    m0, m1, m2, m3 = m0 | d_m0, m1 | d_m1, m2 | d_m2, m3 | d_m3
    a0 = x0 / p0
    a1 = x1 / p1
    a2 = x2 / p2
    a3 = x3 / p3

    return a0, a1, a2, a3, m0, m1, m2, m3


@triton.autotune(
    configs=get_configs(),
    key=['m', 'k'],
)
@triton.jit
def _sparse_semi_structured_from_dense_triton_16(
        dense_ptr,
        sparse_ptr,
        meta_reordered_ptr,
        mask_ptr,
        dense_row_stride,
        sparse_row_stride,
        mask_row_stride,
        dense_col_stride,
        sparse_col_stride,
        mask_col_stride,
        m, k,
        seed,
        BLOCK_SIZE: tl.constexpr,
        PRUNE: tl.constexpr,
        ARRAY_LAYOUT: tl.constexpr,
):
    if ARRAY_LAYOUT == 'row':
        row_idx = tl.program_id(0)
        col_idx = tl.program_id(1) * 16 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 16
        mask = col_idx < k
    elif ARRAY_LAYOUT == 'col':
        row_idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
        col_idx = tl.program_id(1) * 16
        mask = row_idx < m
    dense_40 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 0) * dense_col_stride, mask=mask)
    dense_41 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 1) * dense_col_stride, mask=mask)
    dense_42 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 2) * dense_col_stride, mask=mask)
    dense_43 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 3) * dense_col_stride, mask=mask)
    dense_44 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 4) * dense_col_stride, mask=mask)
    dense_45 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 5) * dense_col_stride, mask=mask)
    dense_46 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 6) * dense_col_stride, mask=mask)
    dense_47 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 7) * dense_col_stride, mask=mask)
    dense_48 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 8) * dense_col_stride, mask=mask)
    dense_49 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 9) * dense_col_stride, mask=mask)
    dense_4A = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 10) * dense_col_stride, mask=mask)
    dense_4B = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 11) * dense_col_stride, mask=mask)
    dense_4C = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 12) * dense_col_stride, mask=mask)
    dense_4D = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 13) * dense_col_stride, mask=mask)
    dense_4E = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 14) * dense_col_stride, mask=mask)
    dense_4F = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 15) * dense_col_stride, mask=mask)

    if PRUNE == 'mse':
        # Triton's bug: can't take abs from bfloat16
        if dense_ptr.type.element_ty == tl.bfloat16:
            (_dense_40, _dense_41, _dense_42, _dense_43, _dense_44, _dense_45, _dense_46, _dense_47, _dense_48,
             _dense_49, _dense_4A, _dense_4B, _dense_4C, _dense_4D, _dense_4E, _dense_4F) = (
                dense_40.to(tl.float32), dense_41.to(tl.float32), dense_42.to(tl.float32), dense_43.to(tl.float32),
                dense_44.to(tl.float32), dense_45.to(tl.float32), dense_46.to(tl.float32), dense_47.to(tl.float32),
                dense_48.to(tl.float32), dense_49.to(tl.float32), dense_4A.to(tl.float32), dense_4B.to(tl.float32),
                dense_4C.to(tl.float32), dense_4D.to(tl.float32), dense_4E.to(tl.float32), dense_4F.to(tl.float32))
        else:
            (_dense_40, _dense_41, _dense_42, _dense_43, _dense_44, _dense_45, _dense_46, _dense_47, _dense_48,
             _dense_49, _dense_4A, _dense_4B, _dense_4C, _dense_4D, _dense_4E, _dense_4F) = (
                dense_40, dense_41, dense_42, dense_43, dense_44, dense_45, dense_46, dense_47, dense_48,
                dense_49, dense_4A, dense_4B, dense_4C, dense_4D, dense_4E, dense_4F)

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_40) > tl.abs(_dense_41), tl.abs(_dense_40) > tl.abs(_dense_42), tl.abs(
            _dense_40) > tl.abs(_dense_43), tl.abs(_dense_41) > tl.abs(_dense_42), tl.abs(_dense_41) > tl.abs(
            _dense_43), tl.abs(
            _dense_42) > tl.abs(_dense_43)
        m0, m1, m2, m3 = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_44) > tl.abs(_dense_45), tl.abs(_dense_44) > tl.abs(_dense_46), tl.abs(
            _dense_44) > tl.abs(_dense_47), tl.abs(_dense_45) > tl.abs(_dense_46), tl.abs(_dense_45) > tl.abs(
            _dense_47), tl.abs(
            _dense_46) > tl.abs(_dense_47)
        m4, m5, m6, m7 = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_48) > tl.abs(_dense_49), tl.abs(_dense_48) > tl.abs(_dense_4A), tl.abs(
            _dense_48) > tl.abs(_dense_4B), tl.abs(_dense_49) > tl.abs(_dense_4A), tl.abs(_dense_49) > tl.abs(
            _dense_4B), tl.abs(
            _dense_4A) > tl.abs(_dense_4B)
        m8, m9, mA, mB = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_4C) > tl.abs(_dense_4D), tl.abs(_dense_4C) > tl.abs(_dense_4E), tl.abs(
            _dense_4C) > tl.abs(_dense_4F), tl.abs(_dense_4D) > tl.abs(_dense_4E), tl.abs(_dense_4D) > tl.abs(
            _dense_4F), tl.abs(
            _dense_4E) > tl.abs(_dense_4F)
        mC, mD, mE, mF = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

    elif PRUNE == 'mask':
        m0 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 0) * mask_col_stride, mask=mask).to(tl.int1)
        m1 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 1) * mask_col_stride, mask=mask).to(tl.int1)
        m2 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 2) * mask_col_stride, mask=mask).to(tl.int1)
        m3 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 3) * mask_col_stride, mask=mask).to(tl.int1)
        m4 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 4) * mask_col_stride, mask=mask).to(tl.int1)
        m5 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 5) * mask_col_stride, mask=mask).to(tl.int1)
        m6 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 6) * mask_col_stride, mask=mask).to(tl.int1)
        m7 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 7) * mask_col_stride, mask=mask).to(tl.int1)
        m8 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 8) * mask_col_stride, mask=mask).to(tl.int1)
        m9 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 9) * mask_col_stride, mask=mask).to(tl.int1)
        mA = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 10) * mask_col_stride, mask=mask).to(tl.int1)
        mB = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 11) * mask_col_stride, mask=mask).to(tl.int1)
        mC = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 12) * mask_col_stride, mask=mask).to(tl.int1)
        mD = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 13) * mask_col_stride, mask=mask).to(tl.int1)
        mE = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 14) * mask_col_stride, mask=mask).to(tl.int1)
        mF = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 15) * mask_col_stride, mask=mask).to(tl.int1)
    elif PRUNE == 'mvue':
        if ARRAY_LAYOUT == 'row':
            seed0 = seed + (tl.program_id(0) + tl.program_id(1) * m) * 2
            seed1 = seed + (tl.program_id(0) + tl.program_id(1) * m) * 2 + 1
        else:
            seed0 = seed + (tl.program_id(0) * k // 16 + tl.program_id(1)) * 2
            seed1 = seed + (tl.program_id(0) * k // 16 + tl.program_id(1)) * 2 + 1
        random0, random1, random2, random3 = tl.rand4x(seed0, tl.arange(0, BLOCK_SIZE), n_rounds=5)
        random4, random5, random6, random7 = tl.rand4x(seed1, tl.arange(0, BLOCK_SIZE), n_rounds=5)

        dense_40, dense_41, dense_42, dense_43, m0, m1, m2, m3 = _MVUE24_approx(dense_40, dense_41, dense_42, dense_43,
                                                                                random0, random1)
        dense_44, dense_45, dense_46, dense_47, m4, m5, m6, m7 = _MVUE24_approx(dense_44, dense_45, dense_46, dense_47,
                                                                                random2, random3)
        dense_48, dense_49, dense_4A, dense_4B, m8, m9, mA, mB = _MVUE24_approx(dense_48, dense_49, dense_4A, dense_4B,
                                                                                random4, random5)
        dense_4C, dense_4D, dense_4E, dense_4F, mC, mD, mE, mF = _MVUE24_approx(dense_4C, dense_4D, dense_4E, dense_4F,
                                                                                random6, random7)
    else:
        m0 = dense_40 != 0
        m1 = dense_41 != 0
        m2 = dense_42 != 0
        m3 = dense_43 != 0
        m4 = dense_44 != 0
        m5 = dense_45 != 0
        m6 = dense_46 != 0
        m7 = dense_47 != 0
        m8 = dense_48 != 0
        m9 = dense_49 != 0
        mA = dense_4A != 0
        mB = dense_4B != 0
        mC = dense_4C != 0
        mD = dense_4D != 0
        mE = dense_4E != 0
        mF = dense_4F != 0

    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = bit1 | ~m2
    bit3 = bit0 | ~m1 | m2
    idxs0 = bit0 | (bit1.to(tl.int64) << 1)
    idxs1 = bit2 | (bit3.to(tl.int64) << 1)
    sparse0 = tl.where(bit1, tl.where(bit0, dense_43, dense_42), tl.where(bit0, dense_41, dense_40))
    sparse1 = tl.where(bit3, tl.where(bit2, dense_43, dense_42), tl.where(bit2, dense_41, dense_40))

    bit4 = ~m4 & m5
    bit5 = ~m4 & ~m5
    bit6 = bit5 | ~m6
    bit7 = bit4 | ~m5 | m6
    idxs2 = bit4 | (bit5.to(tl.int64) << 1)
    idxs3 = bit6 | (bit7.to(tl.int64) << 1)
    sparse2 = tl.where(bit5, tl.where(bit4, dense_47, dense_46), tl.where(bit4, dense_45, dense_44))
    sparse3 = tl.where(bit7, tl.where(bit6, dense_47, dense_46), tl.where(bit6, dense_45, dense_44))

    bit8 = ~m8 & m9
    bit9 = ~m8 & ~m9
    bitA = bit9 | ~mA
    bitB = bit8 | ~m9 | mA
    idxs4 = bit8 | (bit9.to(tl.int64) << 1)
    idxs5 = bitA | (bitB.to(tl.int64) << 1)
    sparse4 = tl.where(bit9, tl.where(bit8, dense_4B, dense_4A), tl.where(bit8, dense_49, dense_48))
    sparse5 = tl.where(bitB, tl.where(bitA, dense_4B, dense_4A), tl.where(bitA, dense_49, dense_48))

    bitC = ~mC & mD
    bitD = ~mC & ~mD
    bitE = bitD | ~mE
    bitF = bitC | ~mD | mE
    idxs6 = bitC | (bitD.to(tl.int64) << 1)
    idxs7 = bitE | (bitF.to(tl.int64) << 1)
    sparse6 = tl.where(bitD, tl.where(bitC, dense_4F, dense_4E), tl.where(bitC, dense_4D, dense_4C))
    sparse7 = tl.where(bitF, tl.where(bitE, dense_4F, dense_4E), tl.where(bitE, dense_4D, dense_4C))

    col_idx = tl.program_id(1) * 8
    if ARRAY_LAYOUT == 'row':
        col_idx = tl.program_id(1) * 8 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 8
        mask = col_idx < k // 2
    else:
        col_idx = tl.program_id(1) * 8
        mask = row_idx < m

    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 0) * sparse_col_stride, sparse0, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 1) * sparse_col_stride, sparse1, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 2) * sparse_col_stride, sparse2, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 3) * sparse_col_stride, sparse3, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 4) * sparse_col_stride, sparse4, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 5) * sparse_col_stride, sparse5, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 6) * sparse_col_stride, sparse6, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 7) * sparse_col_stride, sparse7, mask=mask)

    meta_40 = idxs0 | (idxs1 << 2)
    meta_41 = idxs2 | (idxs3 << 2)
    meta_42 = idxs4 | (idxs5 << 2)
    meta_43 = idxs6 | (idxs7 << 2)
    meta = (
            meta_40
            | (meta_41 << 4)
            | (meta_42 << 8)
            | (meta_43 << 12)
    )

    if ARRAY_LAYOUT == 'row':
        col_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    elif ARRAY_LAYOUT == 'col':
        col_idx = tl.program_id(1)

    group, interweave = 32, 4

    dest_row = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8
    dest_col = col_idx

    topright = ((dest_row % 2 == 0) & (dest_col % 2 == 1)).to(tl.int8)
    bottomleft = ((dest_row % 2 == 1) & (dest_col % 2 == 0)).to(tl.int8)
    dest_row = dest_row + topright - bottomleft
    dest_col = dest_col - topright + bottomleft

    interleave = 2
    cols_maj = dest_col // interleave
    cols_min = dest_col % interleave
    meta_reordered_offsets = (
            cols_maj * m * interleave + dest_row * interleave + cols_min
    )

    if ARRAY_LAYOUT == 'row':
        mask = col_idx < k // 16
    elif ARRAY_LAYOUT == 'col':
        mask = row_idx < m
    tl.store(meta_reordered_ptr + meta_reordered_offsets, meta,
             mask=mask)


@triton.autotune(
    configs=get_configs(),
    key=['m', 'k'],
)
@triton.jit
def _sparse_semi_structured_from_dense_triton_8(
        dense_ptr,
        sparse_ptr,
        meta_reordered_ptr,
        mask_ptr,
        dense_row_stride,
        sparse_row_stride,
        mask_row_stride,
        dense_col_stride,
        sparse_col_stride,
        mask_col_stride,
        m, k,
        seed,
        BLOCK_SIZE: tl.constexpr,
        PRUNE: tl.constexpr,
        ARRAY_LAYOUT: tl.constexpr,
):
    if ARRAY_LAYOUT == 'row':
        row_idx = tl.program_id(0)
        col_idx = tl.program_id(1) * 32 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 32
        mask = col_idx < k
    elif ARRAY_LAYOUT == 'col':
        row_idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
        col_idx = tl.program_id(1) * 32
        mask = row_idx < m
    dense_40 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 0) * dense_col_stride, mask=mask)
    dense_41 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 1) * dense_col_stride, mask=mask)
    dense_42 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 2) * dense_col_stride, mask=mask)
    dense_43 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 3) * dense_col_stride, mask=mask)
    dense_44 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 4) * dense_col_stride, mask=mask)
    dense_45 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 5) * dense_col_stride, mask=mask)
    dense_46 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 6) * dense_col_stride, mask=mask)
    dense_47 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 7) * dense_col_stride, mask=mask)
    dense_48 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 8) * dense_col_stride, mask=mask)
    dense_49 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 9) * dense_col_stride, mask=mask)

    dense_4A = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 10) * dense_col_stride, mask=mask)
    dense_4B = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 11) * dense_col_stride, mask=mask)
    dense_4C = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 12) * dense_col_stride, mask=mask)
    dense_4D = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 13) * dense_col_stride, mask=mask)
    dense_4E = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 14) * dense_col_stride, mask=mask)
    dense_4F = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 15) * dense_col_stride, mask=mask)
    dense_4G = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 16) * dense_col_stride, mask=mask)
    dense_4H = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 17) * dense_col_stride, mask=mask)
    dense_4I = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 18) * dense_col_stride, mask=mask)
    dense_4J = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 19) * dense_col_stride, mask=mask)

    dense_4K = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 20) * dense_col_stride, mask=mask)
    dense_4L = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 21) * dense_col_stride, mask=mask)
    dense_4M = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 22) * dense_col_stride, mask=mask)
    dense_4N = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 23) * dense_col_stride, mask=mask)
    dense_4O = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 24) * dense_col_stride, mask=mask)
    dense_4P = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 25) * dense_col_stride, mask=mask)
    dense_4Q = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 26) * dense_col_stride, mask=mask)
    dense_4R = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 27) * dense_col_stride, mask=mask)
    dense_4S = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 28) * dense_col_stride, mask=mask)
    dense_4T = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 29) * dense_col_stride, mask=mask)

    dense_4U = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 30) * dense_col_stride, mask=mask)
    dense_4V = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 31) * dense_col_stride, mask=mask)

    if PRUNE == 'mse':
        (_dense_40, _dense_41, _dense_42, _dense_43, _dense_44, _dense_45, _dense_46, _dense_47, _dense_48,
         _dense_49, _dense_4A, _dense_4B, _dense_4C, _dense_4D, _dense_4E, _dense_4F) = (
            dense_40, dense_41, dense_42, dense_43, dense_44, dense_45, dense_46, dense_47, dense_48,
            dense_49, dense_4A, dense_4B, dense_4C, dense_4D, dense_4E, dense_4F)

        (_dense_4G, _dense_4H, _dense_4I, _dense_4J, _dense_4K, _dense_4L, _dense_4M, _dense_4N, _dense_4O,
         _dense_4P, _dense_4Q, _dense_4R, _dense_4S, _dense_4T, _dense_4U, _dense_4V) = (
            dense_4G, dense_4H, dense_4I, dense_4J, dense_4K, dense_4L, dense_4M, dense_4N, dense_4O,
            dense_4P, dense_4Q, dense_4R, dense_4S, dense_4T, dense_4U, dense_4V)

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_40) > tl.abs(_dense_41), tl.abs(_dense_40) > tl.abs(_dense_42), tl.abs(
            _dense_40) > tl.abs(_dense_43), tl.abs(_dense_41) > tl.abs(_dense_42), tl.abs(_dense_41) > tl.abs(
            _dense_43), tl.abs(
            _dense_42) > tl.abs(_dense_43)
        m0, m1, m2, m3 = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_44) > tl.abs(_dense_45), tl.abs(_dense_44) > tl.abs(_dense_46), tl.abs(
            _dense_44) > tl.abs(_dense_47), tl.abs(_dense_45) > tl.abs(_dense_46), tl.abs(_dense_45) > tl.abs(
            _dense_47), tl.abs(
            _dense_46) > tl.abs(_dense_47)
        m4, m5, m6, m7 = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_48) > tl.abs(_dense_49), tl.abs(_dense_48) > tl.abs(_dense_4A), tl.abs(
            _dense_48) > tl.abs(_dense_4B), tl.abs(_dense_49) > tl.abs(_dense_4A), tl.abs(_dense_49) > tl.abs(
            _dense_4B), tl.abs(
            _dense_4A) > tl.abs(_dense_4B)
        m8, m9, mA, mB = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_4C) > tl.abs(_dense_4D), tl.abs(_dense_4C) > tl.abs(_dense_4E), tl.abs(
            _dense_4C) > tl.abs(_dense_4F), tl.abs(_dense_4D) > tl.abs(_dense_4E), tl.abs(_dense_4D) > tl.abs(
            _dense_4F), tl.abs(
            _dense_4E) > tl.abs(_dense_4F)
        mC, mD, mE, mF = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_4G) > tl.abs(_dense_4H), tl.abs(_dense_4G) > tl.abs(_dense_4I), tl.abs(
            _dense_4G) > tl.abs(_dense_4J), tl.abs(_dense_4H) > tl.abs(_dense_4I), tl.abs(_dense_4H) > tl.abs(
            _dense_4J), tl.abs(
            _dense_4I) > tl.abs(_dense_4J)
        mG, mH, mI, mJ = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_4K) > tl.abs(_dense_4L), tl.abs(_dense_4K) > tl.abs(_dense_4M), tl.abs(
            _dense_4K) > tl.abs(_dense_4N), tl.abs(_dense_4L) > tl.abs(_dense_4M), tl.abs(_dense_4L) > tl.abs(
            _dense_4N), tl.abs(
            _dense_4M) > tl.abs(_dense_4N)
        mK, mL, mM, mN = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_4O) > tl.abs(_dense_4P), tl.abs(_dense_4O) > tl.abs(_dense_4Q), tl.abs(
            _dense_4O) > tl.abs(_dense_4R), tl.abs(_dense_4P) > tl.abs(_dense_4Q), tl.abs(_dense_4P) > tl.abs(
            _dense_4R), tl.abs(
            _dense_4Q) > tl.abs(_dense_4R)
        mO, mP, mQ, mR = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

        x1, x2, x3, x4, x5, x6 = tl.abs(_dense_4S) > tl.abs(_dense_4T), tl.abs(_dense_4S) > tl.abs(_dense_4U), tl.abs(
            _dense_4S) > tl.abs(_dense_4V), tl.abs(_dense_4T) > tl.abs(_dense_4U), tl.abs(_dense_4T) > tl.abs(
            _dense_4V), tl.abs(
            _dense_4U) > tl.abs(_dense_4V)
        mS, mT, mU, mV = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

    elif PRUNE == 'mask':
        m0 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 0) * mask_col_stride, mask=mask).to(tl.int1)
        m1 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 1) * mask_col_stride, mask=mask).to(tl.int1)
        m2 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 2) * mask_col_stride, mask=mask).to(tl.int1)
        m3 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 3) * mask_col_stride, mask=mask).to(tl.int1)
        m4 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 4) * mask_col_stride, mask=mask).to(tl.int1)
        m5 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 5) * mask_col_stride, mask=mask).to(tl.int1)
        m6 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 6) * mask_col_stride, mask=mask).to(tl.int1)
        m7 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 7) * mask_col_stride, mask=mask).to(tl.int1)
        m8 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 8) * mask_col_stride, mask=mask).to(tl.int1)
        m9 = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 9) * mask_col_stride, mask=mask).to(tl.int1)
        mA = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 10) * mask_col_stride, mask=mask).to(tl.int1)
        mB = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 11) * mask_col_stride, mask=mask).to(tl.int1)
        mC = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 12) * mask_col_stride, mask=mask).to(tl.int1)
        mD = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 13) * mask_col_stride, mask=mask).to(tl.int1)
        mE = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 14) * mask_col_stride, mask=mask).to(tl.int1)
        mF = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 15) * mask_col_stride, mask=mask).to(tl.int1)
        mG = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 16) * mask_col_stride, mask=mask).to(tl.int1)
        mH = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 17) * mask_col_stride, mask=mask).to(tl.int1)
        mI = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 18) * mask_col_stride, mask=mask).to(tl.int1)
        mJ = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 19) * mask_col_stride, mask=mask).to(tl.int1)
        mK = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 20) * mask_col_stride, mask=mask).to(tl.int1)
        mL = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 21) * mask_col_stride, mask=mask).to(tl.int1)
        mM = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 22) * mask_col_stride, mask=mask).to(tl.int1)
        mN = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 23) * mask_col_stride, mask=mask).to(tl.int1)
        mO = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 24) * mask_col_stride, mask=mask).to(tl.int1)
        mP = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 25) * mask_col_stride, mask=mask).to(tl.int1)
        mQ = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 26) * mask_col_stride, mask=mask).to(tl.int1)
        mR = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 27) * mask_col_stride, mask=mask).to(tl.int1)
        mS = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 28) * mask_col_stride, mask=mask).to(tl.int1)
        mT = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 29) * mask_col_stride, mask=mask).to(tl.int1)
        mU = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 30) * mask_col_stride, mask=mask).to(tl.int1)
        mV = tl.load(mask_ptr + row_idx * mask_row_stride + (col_idx + 31) * mask_col_stride, mask=mask).to(tl.int1)
    elif PRUNE == 'mvue':
        if ARRAY_LAYOUT == 'row':
            seed0 = seed + (tl.program_id(0) + tl.program_id(1) * m) * 2
            seed1 = seed + (tl.program_id(0) + tl.program_id(1) * m) * 2 + 1
        else:
            seed0 = seed + (tl.program_id(0) * k // 32 + tl.program_id(1)) * 2
            seed1 = seed + (tl.program_id(0) * k // 32 + tl.program_id(1)) * 2 + 1
        random0, random1, random2, random3 = tl.rand4x(seed0, tl.arange(0, BLOCK_SIZE), n_rounds=5)
        random4, random5, random6, random7 = tl.rand4x(seed1, tl.arange(0, BLOCK_SIZE), n_rounds=5)

        dense_40, dense_41, dense_42, dense_43, m0, m1, m2, m3 = _MVUE24_approx(dense_40, dense_41, dense_42, dense_43,
                                                                                random0, random1)
        dense_44, dense_45, dense_46, dense_47, m4, m5, m6, m7 = _MVUE24_approx(dense_44, dense_45, dense_46, dense_47,
                                                                                random2, random3)
        dense_48, dense_49, dense_4A, dense_4B, m8, m9, mA, mB = _MVUE24_approx(dense_48, dense_49, dense_4A, dense_4B,
                                                                                random4, random5)
        dense_4C, dense_4D, dense_4E, dense_4F, mC, mD, mE, mF = _MVUE24_approx(dense_4C, dense_4D, dense_4E, dense_4F,
                                                                                random6, random7)
    else:
        m0 = dense_40 != 0
        m1 = dense_41 != 0
        m2 = dense_42 != 0
        m3 = dense_43 != 0
        m4 = dense_44 != 0
        m5 = dense_45 != 0
        m6 = dense_46 != 0
        m7 = dense_47 != 0
        m8 = dense_48 != 0
        m9 = dense_49 != 0
        mA = dense_4A != 0
        mB = dense_4B != 0
        mC = dense_4C != 0
        mD = dense_4D != 0
        mE = dense_4E != 0
        mF = dense_4F != 0
        mG = dense_4G != 0
        mH = dense_4H != 0
        mI = dense_4I != 0
        mJ = dense_4J != 0
        mK = dense_4K != 0
        mL = dense_4L != 0
        mM = dense_4M != 0
        mN = dense_4N != 0
        mO = dense_4O != 0
        mP = dense_4P != 0
        mQ = dense_4Q != 0
        mR = dense_4R != 0
        mS = dense_4S != 0
        mT = dense_4T != 0
        mU = dense_4U != 0
        mV = dense_4V != 0

    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = bit1 | ~m2
    bit3 = bit0 | ~m1 | m2
    idxs0 = bit0 | (bit1.to(tl.int64) << 1)
    idxs1 = bit2 | (bit3.to(tl.int64) << 1)
    sparse0 = tl.where(bit1, tl.where(bit0, dense_43, dense_42), tl.where(bit0, dense_41, dense_40))
    sparse1 = tl.where(bit3, tl.where(bit2, dense_43, dense_42), tl.where(bit2, dense_41, dense_40))

    bit4 = ~m4 & m5
    bit5 = ~m4 & ~m5
    bit6 = bit5 | ~m6
    bit7 = bit4 | ~m5 | m6
    idxs2 = bit4 | (bit5.to(tl.int64) << 1)
    idxs3 = bit6 | (bit7.to(tl.int64) << 1)
    sparse2 = tl.where(bit5, tl.where(bit4, dense_47, dense_46), tl.where(bit4, dense_45, dense_44))
    sparse3 = tl.where(bit7, tl.where(bit6, dense_47, dense_46), tl.where(bit6, dense_45, dense_44))

    bit8 = ~m8 & m9
    bit9 = ~m8 & ~m9
    bitA = bit9 | ~mA
    bitB = bit8 | ~m9 | mA
    idxs4 = bit8 | (bit9.to(tl.int64) << 1)
    idxs5 = bitA | (bitB.to(tl.int64) << 1)
    sparse4 = tl.where(bit9, tl.where(bit8, dense_4B, dense_4A), tl.where(bit8, dense_49, dense_48))
    sparse5 = tl.where(bitB, tl.where(bitA, dense_4B, dense_4A), tl.where(bitA, dense_49, dense_48))

    bitC = ~mC & mD
    bitD = ~mC & ~mD
    bitE = bitD | ~mE
    bitF = bitC | ~mD | mE
    idxs6 = bitC | (bitD.to(tl.int64) << 1)
    idxs7 = bitE | (bitF.to(tl.int64) << 1)
    sparse6 = tl.where(bitD, tl.where(bitC, dense_4F, dense_4E), tl.where(bitC, dense_4D, dense_4C))
    sparse7 = tl.where(bitF, tl.where(bitE, dense_4F, dense_4E), tl.where(bitE, dense_4D, dense_4C))

    bitG = ~mG & mH
    bitH = ~mG & ~mH
    bitI = bitH | ~mI
    bitJ = bitG | ~mH | mI
    idxs8 = bitG | (bitH.to(tl.int64) << 1)
    idxs9 = bitI | (bitJ.to(tl.int64) << 1)
    sparse8 = tl.where(bitH, tl.where(bitG, dense_4J, dense_4I), tl.where(bitG, dense_4H, dense_4G))
    sparse9 = tl.where(bitJ, tl.where(bitI, dense_4J, dense_4I), tl.where(bitI, dense_4H, dense_4G))

    bitK = ~mK & mL
    bitL = ~mK & ~mL
    bitM = bitL | ~mM
    bitN = bitK | ~mL | mM
    idxsA = bitK | (bitL.to(tl.int64) << 1)
    idxsB = bitM | (bitN.to(tl.int64) << 1)
    sparseA = tl.where(bitL, tl.where(bitK, dense_4N, dense_4M), tl.where(bitK, dense_4L, dense_4K))
    sparseB = tl.where(bitN, tl.where(bitM, dense_4N, dense_4M), tl.where(bitM, dense_4L, dense_4K))

    bitO = ~mO & mP
    bitP = ~mO & ~mP
    bitQ = bitP | ~mQ
    bitR = bitO | ~mP | mQ
    idxsC = bitO | (bitP.to(tl.int64) << 1)
    idxsD = bitQ | (bitR.to(tl.int64) << 1)
    sparseC = tl.where(bitP, tl.where(bitO, dense_4R, dense_4Q), tl.where(bitO, dense_4P, dense_4O))
    sparseD = tl.where(bitR, tl.where(bitQ, dense_4R, dense_4Q), tl.where(bitQ, dense_4P, dense_4O))

    bitS = ~mS & mT
    bitT = ~mS & ~mT
    bitU = bitT | ~mU
    bitV = bitS | ~mT | mU
    idxsE = bitS | (bitT.to(tl.int64) << 1)
    idxsF = bitU | (bitV.to(tl.int64) << 1)
    sparseE = tl.where(bitT, tl.where(bitS, dense_4V, dense_4U), tl.where(bitS, dense_4T, dense_4S))
    sparseF = tl.where(bitV, tl.where(bitU, dense_4V, dense_4U), tl.where(bitU, dense_4T, dense_4S))

    col_idx = tl.program_id(1) * 16
    if ARRAY_LAYOUT == 'row':
        col_idx = tl.program_id(1) * 16 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 16
        mask = col_idx < k // 2
    else:
        col_idx = tl.program_id(1) * 16
        mask = row_idx < m

    # if FP8 == 'fp8e4b15':
    #     sparse0 = sparse0.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse1 = sparse1.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse2 = sparse2.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse3 = sparse3.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse4 = sparse4.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse5 = sparse5.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse6 = sparse6.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse7 = sparse7.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse8 = sparse8.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparse9 = sparse9.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparseA = sparseA.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparseB = sparseB.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparseC = sparseC.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparseD = sparseD.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparseE = sparseE.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    #     sparseF = sparseF.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    # elif FP8 == 'fp8e4':
    #     sparse0 = sparse0.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse1 = sparse1.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse2 = sparse2.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse3 = sparse3.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse4 = sparse4.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse5 = sparse5.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse6 = sparse6.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse7 = sparse7.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse8 = sparse8.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparse9 = sparse9.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparseA = sparseA.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparseB = sparseB.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparseC = sparseC.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparseD = sparseD.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparseE = sparseE.to(tl.float8e4).to(tl.uint8, bitcast=True)
    #     sparseF = sparseF.to(tl.float8e4).to(tl.uint8, bitcast=True)
    # elif FP8 == 'fp8e5':
    #     sparse0 = sparse0.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse1 = sparse1.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse2 = sparse2.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse3 = sparse3.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse4 = sparse4.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse5 = sparse5.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse6 = sparse6.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse7 = sparse7.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse8 = sparse8.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparse9 = sparse9.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparseA = sparseA.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparseB = sparseB.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparseC = sparseC.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparseD = sparseD.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparseE = sparseE.to(tl.float8e5).to(tl.uint8, bitcast=True)
    #     sparseF = sparseF.to(tl.float8e5).to(tl.uint8, bitcast=True)

    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 0) * sparse_col_stride, sparse0, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 1) * sparse_col_stride, sparse1, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 2) * sparse_col_stride, sparse2, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 3) * sparse_col_stride, sparse3, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 4) * sparse_col_stride, sparse4, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 5) * sparse_col_stride, sparse5, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 6) * sparse_col_stride, sparse6, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 7) * sparse_col_stride, sparse7, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 8) * sparse_col_stride, sparse8, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 9) * sparse_col_stride, sparse9, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 10) * sparse_col_stride, sparseA, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 11) * sparse_col_stride, sparseB, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 12) * sparse_col_stride, sparseC, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 13) * sparse_col_stride, sparseD, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 14) * sparse_col_stride, sparseE, mask=mask)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 15) * sparse_col_stride, sparseF, mask=mask)

    meta_40 = idxs0 | (idxs1 << 2)
    meta_41 = idxs2 | (idxs3 << 2)
    meta_42 = idxs4 | (idxs5 << 2)
    meta_43 = idxs6 | (idxs7 << 2)
    meta_44 = idxs8 | (idxs9 << 2)
    meta_45 = idxsA | (idxsB << 2)
    meta_46 = idxsC | (idxsD << 2)
    meta_47 = idxsE | (idxsF << 2)
    meta = (
            meta_40
            | (meta_41 << 4)
            | (meta_42 << 8)
            | (meta_43 << 12)
            | (meta_44 << 16)
            | (meta_45 << 20)
            | (meta_46 << 24)
            | (meta_47 << 28)
    )

    if ARRAY_LAYOUT == 'row':
        col_idx = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    elif ARRAY_LAYOUT == 'col':
        col_idx = tl.program_id(1)

    group, interweave = 16, 2

    dest_row = row_idx // group * group + (row_idx % 8) * interweave + (row_idx % group) // 8
    dest_col = col_idx

    topright = ((dest_row % 2 == 0) & (dest_col % 2 == 1)).to(tl.int8)
    bottomleft = ((dest_row % 2 == 1) & (dest_col % 2 == 0)).to(tl.int8)
    dest_row = dest_row + topright - bottomleft
    dest_col = dest_col - topright + bottomleft

    interleave = 2
    cols_maj = dest_col // interleave
    cols_min = dest_col % interleave
    meta_reordered_offsets = (
            cols_maj * m * interleave + dest_row * interleave + cols_min
    )

    if ARRAY_LAYOUT == 'row':
        mask = col_idx < k // 32
    elif ARRAY_LAYOUT == 'col':
        mask = row_idx < m
    tl.store(meta_reordered_ptr + meta_reordered_offsets, meta,
             mask=mask)


def sparse_semi_structured_from_dense_triton(dense, sparse, meta, MVUE24: bool = False,
                                             mask: Optional[Tensor] = None):
    m, k = dense.shape
    device = dense.device
    seed = random.randint(0, 2 ** 31 - 1)

    if mask is not None:
        prune = 'mask'
    elif MVUE24:
        prune = 'mvue'
    else:
        prune = 'mse'

    quadbits_per_meta_elem = meta.dtype.itemsize * 8 // 4
    meta_cols_width = 4 * quadbits_per_meta_elem

    row_stride, col_stride = dense.stride()
    if row_stride > col_stride:
        array_layout = 'row'
        grid = lambda META: (m, triton.cdiv(k, meta_cols_width * META['BLOCK_SIZE']))
    else:
        array_layout = 'col'
        grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE']), k // meta_cols_width,)

    if meta.dtype == torch.int32:
        func = _sparse_semi_structured_from_dense_triton_8
    else:
        func = _sparse_semi_structured_from_dense_triton_16

    if dense.stride(0) < dense.stride(1) and sparse.stride(0) > sparse.stride(1):
        array_layout = 'row'
        grid = lambda META: (m, triton.cdiv(k, meta_cols_width * META['BLOCK_SIZE']))

    # if fp8 is not None:
    #     workspace = torch.empty_like(sparse, dtype=dense.dtype)
    # else:
    #     workspace = None

    func[grid](
        dense,
        sparse,
        meta,
        mask,
        dense.stride(0),
        sparse.stride(0),
        mask.stride(0) if mask is not None else None,
        dense.stride(1),
        sparse.stride(1),
        mask.stride(1) if mask is not None else None,
        m, k,
        seed,
        PRUNE=prune,
        ARRAY_LAYOUT=array_layout,
    )

    # if fp8 is not None:
    #     n_elements = sparse.numel()
    #     grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    #     _cast[grid](sparse, workspace, n_elements, FP8=fp8)

    return (sparse, meta)


# @triton.autotune(
#     configs=get_cast_configs(),
#     key=['n_elements'],
# )
# @triton.jit
# def _cast(
#         sparse_ptr,
#         workspace_ptr,
#         n_elements,
#         BLOCK_SIZE: tl.constexpr,
#         FP8: tl.constexpr
# ):
#     pid = tl.program_id(axis=0)
#     block_start = pid * BLOCK_SIZE
#     offsets = block_start + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < n_elements
#
#     sparse = tl.load(workspace_ptr + offsets, mask=mask)
#     if FP8 == 'fp8e4b15':
#         sparse = sparse.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
#     elif FP8 == 'fp8e4':
#         sparse = sparse.to(tl.float8e4).to(tl.uint8, bitcast=True)
#     elif FP8 == 'fp8e5':
#         sparse = sparse.to(tl.float8e5).to(tl.uint8, bitcast=True)
#
#     tl.store(sparse_ptr + offsets, sparse, mask=mask)


@triton.jit
def _sparse_semi_structured_to_dense_kernel(
        sparse_ptr,
        meta_reordered_ptr,
        dense_ptr,
        m, k,  # dense.shape
        BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    group, interweave = 32, 4
    dest_row = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8
    if dest_row % 2 == 0:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 + tl.arange(0, BLOCK_SIZE // 16) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        meta = tl.load(meta_reordered_ptr + index, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                       other=-float('inf'))  # shape=k//16
    else:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 - (
                tl.arange(0, BLOCK_SIZE // 16) + 1) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2 + 1
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        meta = tl.load(meta_reordered_ptr + index, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                       other=-float('inf'))  # shape=k//16

    meta_20 = (meta & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16))
    meta_21 = ((meta >> 2) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16))
    meta_22 = ((meta >> 4) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 4)
    meta_23 = ((meta >> 6) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 4)
    meta_24 = ((meta >> 8) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 8)
    meta_25 = ((meta >> 10) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 8)
    meta_26 = ((meta >> 12) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 12)
    meta_27 = ((meta >> 14) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 12)

    row0 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16),
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row1 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 1,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row2 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 2,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row3 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 3,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row4 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 4,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row5 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 5,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row6 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 6,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row7 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 7,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))

    tl.store(dense_ptr + meta_20, row0, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_21, row1, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_22, row2, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_23, row3, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_24, row4, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_25, row5, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_26, row6, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_27, row7, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)


def _sparse_semi_structured_to_dense_triton(sparse, meta_reordered):
    assert sparse.is_contiguous()
    assert meta_reordered.is_contiguous()
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    m, k = sparse.shape[0], sparse.shape[1] * 2
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype is not torch.int16:
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")

    BLOCK_SIZE = triton.next_power_of_2(k)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # num_warps = 2
    # if BLOCK_SIZE >= 2048:
    #     num_warps = 4
    # if BLOCK_SIZE >= 4096:
    #     num_warps = 8
    # if BLOCK_SIZE >= 8192:
    #     num_warps = 16

    dense = torch.zeros((m, k), dtype=sparse.dtype, device=device)
    _sparse_semi_structured_to_dense_kernel[(m,)](
        sparse,
        meta_reordered,
        dense,
        m, k,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dense
