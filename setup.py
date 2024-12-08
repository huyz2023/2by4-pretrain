#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

extra_compile_args = {'cxx': ['-O2']}
extra_compile_args['nvcc'] = ['-O3',
                              '-I./cutlass/include',
                              '-gencode', 'arch=compute_80,code=sm_80',
                              '-gencode', 'arch=compute_80,code=compute_80',
                              '-gencode', 'arch=compute_89,code=sm_89',
                              '-gencode', 'arch=compute_89,code=compute_89',
                              ]

strided_batched_gemm = CUDAExtension(
    name='sparse.cutlass',
    sources=['sparse/csrc/mySparseSemiStructuredLinear_cuda.cu'],
    extra_compile_args=extra_compile_args
)

setup(
    name='sparse',
    version='0.2.0',
    description='2by4-spMM Toolkit',
    packages=find_packages(),
    ext_modules=[strided_batched_gemm],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
)
