import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

# x = torch.randn(128, 128, device='cuda').to(torch.float8_e5m2).half()
# print(x[:4,:4])
# y = to_sparse_semi_structured(x, dtype=torch.float8_e5m2)
# z = y.to_dense()
# print(z[:4,:4])


# x = torch.randn(128, 128, device='cuda').half()
# y = to_sparse_semi_structured(x, dtype=torch.float8_e4m3fn)
# y_ = y.to_dense()
# z = torch.randn(128, 128, device='cuda').to(torch.float8_e4m3fn).t()
#
#
#
# w = torch.mm(y, z)
# w_ = torch.mm(y_, z.float())
# print(w[:4, :4])
# print(w_[:4, :4])

A = torch.randn(128, 128, device='cuda:0', dtype=torch.half)
A_sparse = to_sparse_semi_structured(A)
z = torch.randn(128, 128, device='cuda').half()
w = torch.mm(y, z)
