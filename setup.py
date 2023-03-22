# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension

extra_compile_args = {'cxx': ['-O3'],
                      'nvcc': ['-O3', '--compiler-options', "'-fPIC'"],
                      'cpp': ['-O3', '-Wno-reorder']}
include_dirs_r = ['/usr/local/cuda/include']
library_dirs_r = ['/usr/local/cuda/lib64']
libraries = ['cudart']
define_macros = [('WITH_CUDA', None)]
undef_macros = ['NDEBUG']
optional = True

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "multiplexer", "layers", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        # if True:
        # extension = CUDAExtension
        
        extension = CUDAExtension(name='multiplexer._C',
                    sources=['src/multiplexer.cpp',
                             'src/multiplexer_cuda.cu'],
                    extra_compile_args=extra_compile_args,
                    include_dirs=include_dirs_r,
                    library_dirs=library_dirs_r,
                    libraries=libraries,
                    define_macros=define_macros,
                    undef_macros=undef_macros,
                    optional=optional,
                    version='11.8')
        
        
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "multiplexer._C",
            sources,
            include_dirs=include_dirs_r,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="Multiplexer",
    version="1.0",
    author="raj15019",
    url="https://github.com/raj15019/MultiplexedOCR",
    description="Multiplexed OCR",
    packages=find_packages(
        exclude=(
            "configs",
            "examples",
            "test",
        )
    ),
    # install_requires=requirements,
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)}
)
