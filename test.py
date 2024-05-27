import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from sparse import to_sparse_semi_structured, SparseSemiStructuredTensor


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[768, 1024, 2048, 4096, 8192],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    SparseSemiStructuredTensor._FORCE_CUTLASS = False
    N = 64
    n = 2048
    x = torch.rand(size, N * n, device='cuda', dtype=torch.half)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: F.relu(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: to_sparse_semi_structured(x, dtype=torch.half),
                                                     quantiles=quantiles)
    gbps = lambda ms: N * n * size * 2 * 2 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=False)
