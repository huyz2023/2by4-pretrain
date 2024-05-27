import warnings
from collections import namedtuple
from typing import Any, Optional

import torch

import sparse

__all__ = [
    "SparseSemiStructuredTensor",
    "to_sparse_semi_structured",
]

# _SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
#     "_SEMI_STRUCTURED_SPARSE_CONFIG", "compression_factor min_rows min_cols"
# )
# _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG = {
#     torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(9, 32, 64),
#     torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(9, 32, 64),
# }
_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG", "compression_factor sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols"
)
_DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG = {
    torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(10, 32, 128, 16, 16),
    torch.float8_e5m2: _SEMI_STRUCTURED_SPARSE_CONFIG(10, 32, 128, 16, 16),
    torch.float8_e4m3fn: _SEMI_STRUCTURED_SPARSE_CONFIG(10, 32, 128, 16, 16),
    torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(9, 32, 64, 8, 8),
    torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(9, 32, 64, 8, 8)
}


class SparseSemiStructuredTensor(torch.Tensor):
    """This class implementes semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    Currently, this class supports 2:4 sparsity for int8, float16 and bfloat16 dtypes.
    We also support 1:2 sparsity for float32 dtype.

    This subclass stores the dense tensor in a compressed form by only storing the specified elements and corresponding metadata.

    The subclass supports two backend, either CUTLASS or cuSPASRELt.

    The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:

    compressed tensor = [ specified elements of original tensor | metadata ]

    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata.

    For CUTLASS backend, elements of original tensor and metadata are kept in separate tensors.

    When _FORCE_CUTLASS is set, or when cuSPARSELt is not available, this subclass calls into _sparse_semi_structured_linear
    and sparse_semi_structured_from_dense for conversion to the compressed format.

    When PyTorch is compiled with cuSPARSELt support, this subclass will call into _cslt_sparse_mm for sparse mm and
    _cslt_compress to convert into the compressed format.
    """

    _FUSE_TRANSPOSE = False
    _FORCE_CUTLASS = True
    _PROTOTYPE_WARNING_SHOWN = False

    @staticmethod
    def __new__(
            cls,
            original_tensor: Optional[torch.Tensor],
            original_shape: Optional[torch.Size] = None,
            compressed_tensor_cusparselt: Optional[torch.Tensor] = None,
            sparse_tensor_cutlass: Optional[torch.Tensor] = None,
            meta_tensor_cutlass: Optional[torch.Tensor] = None,
            transposed: bool = False,
            MVUE24: bool = False,
            mask: Optional[torch.Tensor] = None,
            dtype: Optional[torch.dtype] = None,
    ):
        """
        Create a new instance of the class.

        When original_tensor is passed in, we compress it and store the compresed representation.
        We can also create new instance of the class from the compressed representation without the original tensor.

        Args:
            original_tensor: The original dense tensor, or None, if we have already compressed the tensor.
            original_shape: The shape of the original dense tensor
            compressed_tensor_cusparselt: For cuSPARSELt backend, a flattened tensor to store the specified elements and metadata.
            sparse_tensor_cutlass: For CUTLASS backend, tensor to store the speficied elements.
            meta_tensor_cutlass: For CUTLASS backend, tensor to store metadata.
            transposed: Whether the tensor is transposed or not.

        Returns:
            torch.Tensor: A torch.Tensor wrapper subclass.

        Raises:
            ValueError: If all of the tensor arguments are None.

        """
        assert compressed_tensor_cusparselt is None or (sparse_tensor_cutlass is None and meta_tensor_cutlass is None)

        if not cls._PROTOTYPE_WARNING_SHOWN:
            warnings.warn(
                (
                    "The PyTorch API of SparseSemiStructuredTensor is in prototype stage "
                    "and will change in the near future. Please open a Github issue "
                    "for features requests and see our documentation on the torch.sparse "
                    "module for further information about the project."
                ),
                UserWarning,
            )
            cls._PROTOTYPE_WARNING_SHOWN = True

        if original_tensor is not None:
            previous_tensor = original_tensor
            original_shape = original_tensor.shape
        elif compressed_tensor_cusparselt is not None:
            previous_tensor = compressed_tensor_cusparselt
        elif sparse_tensor_cutlass is not None:
            previous_tensor = sparse_tensor_cutlass
        else:
            raise ValueError("All of the tensor arguments are None!")

        kwargs = {}
        kwargs["device"] = previous_tensor.device  # type: ignore[assignment]
        kwargs["dtype"] = previous_tensor.dtype if dtype is None else dtype  # type: ignore[assignment]
        kwargs["layout"] = previous_tensor.layout  # type: ignore[assignment]
        kwargs["requires_grad"] = False  # type: ignore[assignment]

        return torch.Tensor._make_wrapper_subclass(cls, original_shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
            self,
            original_tensor: Optional[torch.Tensor],
            original_shape: Optional[torch.Size] = None,
            compressed_tensor_cusparselt: Optional[torch.Tensor] = None,
            sparse_tensor_cutlass: Optional[torch.Tensor] = None,
            meta_tensor_cutlass: Optional[torch.Tensor] = None,
            transposed: bool = False,
            MVUE24: bool = False,
            mask: Optional[torch.Tensor] = None,
            dtype: Optional[torch.dtype] = torch.float16
    ) -> None:
        """SparseSemiStructuredTensor constructor.

        Args:
            original_tensor: The original dense tensor, or None, if we have already compressed the tensor.
            original_shape: The shape of the original dense tensor
            compressed_tensor_cusparselt: For cuSPARSELt backend, a flattened tensor to store the specified elements and metadata.
            sparse_tensor_cutlass: For CUTLASS backend, tensor to store the speficied elements.
            meta_tensor_cutlass: For CUTLASS backend, tensor to store metadata.
            transposed: Whether the tensor is transposed or not.

        Returns:
            None

        Raises:
            RuntimeError: If original_tensor is not a supported dtype, dim, shape, or device.
        """
        # if original tensor is passed in, we need to compress it and store the compressed representation.
        if original_tensor is not None:
            # TODO right now we have unified checks and constraints for cuSPARSELt and CUTLASS, these are not actually the same.
            # We should consolidate similar checks here and leave backend specific checks like shape in the op implementation.

            # check device
            if not original_tensor.is_cuda:
                raise RuntimeError(
                    f"Error original_tensor.device= {original_tensor.device} is not supported! "
                    "Only CUDA tensors are currently supported."
                )

            # check dim
            if original_tensor.dim() != 2:
                raise RuntimeError(
                    f"Error original_tensor.dim = {original_tensor.dim()} is not supported! "
                    "Only 2d tensors are currently supported."
                )

            # check dtype
            if dtype not in _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG:
                raise RuntimeError(
                    f"Error dtype {dtype} is not a supported dtype! "
                    "dtype must be one of: {_DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG}"
                )

            # check shape
            m, n = original_tensor.shape
            min_rows = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[
                dtype
            ].sparse_min_rows
            min_cols = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[
                dtype
            ].sparse_min_cols
            if m < min_rows or m % min_rows or n < min_cols or n % min_cols:
                # TODO in the future we can add in padding to support sparse dimensions that aren't perfect multiples
                raise RuntimeError(
                    f"Error original_tensor.shape {original_tensor.shape} is not supported! "
                    f"Both dimensions must be larger or equal than and a multiple of ({min_rows}, {min_cols})"
                )

            compressed_tensor_cusparselt = None
            sparse_tensor_cutlass = None
            meta_tensor_cutlass = None
            if self._FORCE_CUTLASS:
                # check availability
                if mask is not None and MVUE24:
                    raise RuntimeError(
                        f"Error using MVUE24 with mask is not supported! "
                    )

                # This code calculates the size of the compressed tensor.
                # compression factor is different based on dtype it's given by the formula below for 2:4 sparsity:
                # compression_factor = 1/2 + 1/bitwidth(dtype)
                original_size = original_tensor.nelement()
                compression_factor = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[
                    dtype
                ].compression_factor
                compressed_size = original_size * compression_factor // 16

                compressed_tensor = torch.empty(
                    (compressed_size,),
                    dtype=dtype,
                    device=original_tensor.device,
                )
                row_stride, col_stride = original_tensor.stride()

                from ._semi_structured_conversions import (
                    sparse_semi_structured_from_dense_triton
                )
                if row_stride > col_stride:
                    indices_dtype = SparseSemiStructuredTensor.__get_indices_dtype(dtype)
                    sparse_tensor_cutlass, meta_tensor_cutlass = compressed_tensor[: m * n // 2].view(m,
                                                                                                      -1), compressed_tensor[
                                                                                                           m * n // 2:].view(
                        indices_dtype).view(m, -1)

                else:
                    indices_dtype = SparseSemiStructuredTensor.__get_indices_dtype(dtype)
                    sparse_tensor_cutlass, meta_tensor_cutlass = compressed_tensor[: m * n // 2].view(-1,
                                                                                                      m).t(), compressed_tensor[
                                                                                                              m * n // 2:].view(
                        indices_dtype).view(m, -1)
                sparse_semi_structured_from_dense_triton(original_tensor, sparse_tensor_cutlass,
                                                         meta_tensor_cutlass, fp8, MVUE24=MVUE24, mask=mask)

            else:
                # # use cuSPARSELt
                # compressed_tensor_cusparselt = torch._cslt_compress(original_tensor)

                # check availability
                if mask is not None and MVUE24:
                    raise RuntimeError(
                        f"Error using MVUE24 with mask is not supported! "
                    )

                # This code calculates the size of the compressed tensor.
                # compression factor is different based on dtype it's given by the formula below for 2:4 sparsity:
                # compression_factor = 1/2 + 1/bitwidth(dtype)
                original_size = original_tensor.nelement()
                compression_factor = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[
                    dtype
                ].compression_factor
                compressed_size = original_size * compression_factor // 16

                compressed_tensor_cusparselt = torch.empty(
                    (compressed_size,),
                    dtype=dtype,
                    device=original_tensor.device,
                )

                from ._semi_structured_conversions import (
                    sparse_semi_structured_from_dense_triton
                )

                indices_dtype = SparseSemiStructuredTensor.__get_indices_dtype(dtype)
                sparse_tensor, meta_tensor = compressed_tensor_cusparselt[: m * n // 2].view(m,
                                                                                             -1), compressed_tensor_cusparselt[
                                                                                                  m * n // 2:].view(
                    indices_dtype).view(m, -1)

                sparse_semi_structured_from_dense_triton(original_tensor, sparse_tensor,
                                                         meta_tensor, MVUE24=MVUE24, mask=mask)

        # set values
        self.original_tensor = None
        self.compressed_tensor_cusparselt = compressed_tensor_cusparselt
        self.sparse_tensor_cutlass = sparse_tensor_cutlass
        self.meta_tensor_cutlass = meta_tensor_cutlass
        self.transposed = transposed
        self.original_shape = original_shape

    def __tensor_flatten__(self):
        if self.compressed_tensor_cusparselt is None:
            return ['sparse_tensor_cutlass', 'meta_tensor_cutlass'], (self.original_shape, self.transposed)
        else:
            return ['compressed_tensor_cusparselt'], (self.original_shape, self.transposed)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        original_shape, transposed = meta

        if len(inner_tensors) == 2:
            sparse_tensor_cutlass = inner_tensors['sparse_tensor_cutlass']
            meta_tensor_cutlass = inner_tensors['meta_tensor_cutlass']
            compressed_tensor_cusparselt = None
        elif len(inner_tensors) == 1:
            sparse_tensor_cutlass = None
            meta_tensor_cutlass = None
            compressed_tensor_cusparselt = inner_tensors['compressed_tensor_cusparselt']
        else:
            raise RuntimeError(f"Expected 1 or 2 inner tensors but got {len(inner_tensors)}")

        return SparseSemiStructuredTensor(
            None,
            original_shape=original_shape,
            compressed_tensor_cusparselt=compressed_tensor_cusparselt,
            sparse_tensor_cutlass=sparse_tensor_cutlass,
            meta_tensor_cutlass=meta_tensor_cutlass,
            transposed=transposed,
        )

    @staticmethod
    def __get_indices_dtype(values_dtype):
        if values_dtype in (torch.int8, torch.float8_e5m2, torch.float8_e4m3fn):
            return torch.int32
        elif values_dtype in (torch.float16, torch.bfloat16):
            return torch.int16
        else:
            raise RuntimeError(f"Datatype {values_dtype}  is not supported!")
        return None

    def __repr__(self) -> str:  # type: ignore[override]
        """Return string representation of SparseSemiStructuredTensor

        Returns:
            str: String representation

        Raises:
            None
        """
        return (
            f"SparseSemiStructuredTensor(shape={self.shape}, "
            f"transposed={self.transposed}"
            f"values={self.values()}"
            f"metadata={self.indices()})"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    def _pad_tensor_for_matmul(self, original_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
        # only 2d matmul
        assert original_tensor.dim() == 2

        # check shape
        m, n = original_tensor.shape
        min_rows = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].dense_min_rows
        min_cols = _DTYPE_TO_SEMI_STRUCTURED_SPARSE_CONFIG[original_tensor.dtype].dense_min_cols
        to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
        to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
        if to_pad_m or to_pad_n:
            return torch.nn.functional.pad(original_tensor, (0, to_pad_n, 0, to_pad_m))
        else:
            return original_tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:
        """Overload __torch_dispatch__ to use torch._sparse_semi_structured_linear.

        `torch.structured_sparse_linear` uses accelerated sparse CUTLASS kernels.
        In the future we plan to also add in support for cuSPARSELt kernels.

        Args:
            func: The function being dispatched.
            types: The types of the arguments.
            args: The arguments passed to the function.
            kwargs: The keyword arguments passed to the function.

        Returns:
            Any: The result of the dispatched operation.

        Raises:
            NotImplementedError: If the dispatched operation is not implemented.
        """
        # Since this code runs below autograd, a detach corresponds to only returning a new object
        if func is torch.ops.aten.detach.default:
            return SparseSemiStructuredTensor(
                args[0].original_tensor,
                original_shape=args[0].shape,
                compressed_tensor_cusparselt=args[0].compressed_tensor_cusparselt,
                sparse_tensor_cutlass=args[0].sparse_tensor_cutlass,
                meta_tensor_cutlass=args[0].meta_tensor_cutlass,
                transposed=args[0].transposed,
            )

        # Because we cannot go from the compressed representation back to the dense representation currently,
        # we just keep track of how many times we have been transposed. Depending on whether the sparse matrix
        # is the first or second argument, we expect an even / odd number of calls to transpose respectively.
        if func is torch.ops.aten.t.default:
            return SparseSemiStructuredTensor(
                args[0].original_tensor,
                # transpose shape
                original_shape=torch.Size([args[0].shape[1], args[0].shape[0]]),
                compressed_tensor_cusparselt=args[0].compressed_tensor_cusparselt,
                sparse_tensor_cutlass=args[0].sparse_tensor_cutlass,
                meta_tensor_cutlass=args[0].meta_tensor_cutlass,
                transposed=not args[0].transposed,
            )

        # handle addmm
        if func is torch.ops.aten.addmm.default:
            bias, input_A, input_B = args

            # Currently, we only support the first matrix being sparse for addmm/mm in cuSPARSELT and CUTLASS.
            # CUTLASS only supports the first input to be sparse for a given matmul.
            # cuSPARSELt does not have this limitation, although our implementation is only for sparse first.

            # We support second matrix sparse matmul by taking advantage of some transpose properties:
            # This is also why we want an odd number of transposed for second matrix sparse vs an even number
            # of transpose calss for first matrix sparse.
            # F.linear(x) = addmm(bias, input, weight.t()) = b + xW' = (b + xW')''
            #        = (W''x' + b')' = (Wx' + b')' = addmm(bias.T, weight, input).T
            if isinstance(input_B, cls) and input_B.transposed:
                row, col = input_A.shape
                input_A_padded = input_B._pad_tensor_for_matmul(input_A)

                if input_B.compressed_tensor_cusparselt is None:
                    assert input_B.sparse_tensor_cutlass is not None and input_B.meta_tensor_cutlass is not None
                    res = sparse._my_sparse_semi_structured_linear(
                        input_A_padded,
                        input_B.sparse_tensor_cutlass,
                        input_B.meta_tensor_cutlass,
                        bias=bias
                    )
                else:
                    res = torch._cslt_sparse_mm(
                        input_B.compressed_tensor_cusparselt,
                        input_A_padded.t(),
                        bias=bias,  # type: ignore[arg-type]
                        transpose_result=cls._FUSE_TRANSPOSE
                    )
                    res = res if cls._FUSE_TRANSPOSE else res.t()
                return res[:row, :]

        # handle mm
        if func is torch.ops.aten.mm.default:
            input_A, input_B = args

            # first element sparse
            if isinstance(input_A, cls) and not input_A.transposed:
                row, col = input_B.shape
                input_B_padded = input_A._pad_tensor_for_matmul(input_B)

                if input_A.compressed_tensor_cusparselt is None:
                    assert input_A.sparse_tensor_cutlass is not None and input_A.meta_tensor_cutlass is not None
                    res = sparse._my_sparse_semi_structured_linear(
                        input_B_padded.t(),
                        input_A.sparse_tensor_cutlass,
                        input_A.meta_tensor_cutlass
                    ).t()
                else:
                    res = torch._cslt_sparse_mm(
                        input_A.compressed_tensor_cusparselt,
                        input_B_padded,
                        bias=None  # type: ignore[arg-type]
                    )
                return res[:, :col]

            # second element sparse
            elif isinstance(input_B, cls) and input_B.transposed:
                row, col = input_A.shape
                input_A_padded = input_B._pad_tensor_for_matmul(input_A)

                if input_B.compressed_tensor_cusparselt is None:
                    assert input_B.sparse_tensor_cutlass is not None and input_B.meta_tensor_cutlass is not None
                    res = sparse._my_sparse_semi_structured_linear(
                        input_A_padded,
                        input_B.sparse_tensor_cutlass,
                        input_B.meta_tensor_cutlass,
                    )
                else:
                    res = torch._cslt_sparse_mm(
                        input_B.compressed_tensor_cusparselt,
                        input_A_padded.t(),
                        bias=None,  # type: ignore[arg-type]
                        transpose_result=cls._FUSE_TRANSPOSE
                    )
                    res = res if cls._FUSE_TRANSPOSE else res.t()
                return res[:row, :]

        # When torch is run with inference mode, pytorch does not decompose torch.ops.aten.linear into a .t() and addmm(),
        # so we must match the aten.linear op. In this case, we need to explicitly handle collapsing to 2d matmul
        # TODO see if there's a way to force pytorch to decompose the op so we don't have to handle this here.
        if func is torch.ops.aten.linear.default:
            input_tensor, weight, bias = args
            # squash input_tensor to 2d
            shape = input_tensor.shape
            input_tensor_2d = input_tensor.view(-1, shape[-1])
            row, col = input_tensor_2d.shape
            # this is a noop if already padded
            input_tensor_2d_padded = weight._pad_tensor_for_matmul(input_tensor_2d)

            if isinstance(weight, cls):
                if weight.compressed_tensor_cusparselt is None:
                    assert weight.sparse_tensor_cutlass is not None and weight.meta_tensor_cutlass is not None
                    res = sparse._my_sparse_semi_structured_linear(
                        input_tensor_2d_padded,
                        weight.sparse_tensor_cutlass,
                        weight.meta_tensor_cutlass,
                        bias=bias
                    )
                else:
                    res = torch._cslt_sparse_mm(
                        weight.compressed_tensor_cusparselt,  # type: ignore[arg-type]
                        input_tensor_2d_padded.t(),
                        bias=bias,
                        transpose_result=cls._FUSE_TRANSPOSE
                    )
                    res = res if cls._FUSE_TRANSPOSE else res.t()
                return res[:row, :].view(*shape[:-1], -1)

        # handle values
        if func is torch.ops.aten.values.default:
            if args[0].compressed_tensor_cusparselt is None:
                return args[0].sparse_tensor_cutlass.detach()
            else:
                m, k = args[0].shape
                num_kept_elements = m * k // 2
                return args[0].compressed_tensor_cusparselt[:num_kept_elements].view(m, k // 2)

        # handle indices
        if func is torch.ops.aten.indices.default:
            if args[0].compressed_tensor_cusparselt is None:
                return args[0].meta_tensor_cutlass
            else:
                m, k = args[0].shape
                num_kept_elements = m * k // 2
                metadata = args[0].compressed_tensor_cusparselt[num_kept_elements:].view(m, -1)
                indices_dtype = SparseSemiStructuredTensor.__get_indices_dtype(
                    args[0].dtype
                )
                return metadata.view(indices_dtype)

        error_string = "\n".join(
            [f"func {func} with args: "]
            + [f"arg{i}: {arg}" for i, arg in enumerate(args)]
        )
        raise NotImplementedError(error_string)

    def to_dense(self):
        if self.compressed_tensor_cusparselt is not None:
            #     raise RuntimeError("Converting to dense is not yet supported by cuSPARSELt backend!")
            from ._semi_structured_conversions import (
                sparse_semi_structured_to_dense,
            )
            return sparse_semi_structured_to_dense(
                self.values(),
                self.indices(),
            )

        from ._semi_structured_conversions import (
            sparse_semi_structured_to_dense,
        )

        return sparse_semi_structured_to_dense(
            self.sparse_tensor_cutlass.contiguous(),
            self.meta_tensor_cutlass.contiguous(),
        )


def to_sparse_semi_structured(
        original_tensor: torch.Tensor,
        transposed: bool = False,
        MVUE24: bool = False,
        mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float16,
) -> SparseSemiStructuredTensor:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of a block size given the dtype

    - torch.float16  (r, c) must be >= and a multiple of 64
    - torch.int8     (r, c) must be >= and a multiple of 128

    Args:
        original_tensor (Tensor): the dense tensor to convert
        transposed (bool, optional): whether the dense tensor is transposed

    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor

    Raises:
        None
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A)
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]), transposed=False, values=tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
            metadata=tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0',
       dtype=torch.int16))
    """
    return SparseSemiStructuredTensor(
        original_tensor, original_shape=original_tensor.shape, transposed=transposed,
        MVUE24=MVUE24, mask=mask, dtype=dtype)
