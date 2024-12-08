import sparse.cutlass


def sparse_semi_structured_linear(input, weight, meta, bias_opt=None, activation_opt=None):
    return sparse.cutlass.my_sparse_semi_structured_linear(input, weight, meta, bias_opt, activation_opt)
