# Copyright (c) 2025, Tencent Inc. All rights reserved.
import os
import pathlib
import shutil
import subprocess
import warnings

import torch
from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


def load(compile_rotary=True,
         compile_xentropy=True,
         force_compile=False,
         verbose=True):
    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    if torch.version.hip is None:
        _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(
            cpp_extension.CUDA_HOME)
        if int(bare_metal_major) >= 11:
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_80,code=sm_80')
            if int(bare_metal_minor) >= 1:
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_86,code=sm_86')
            if int(bare_metal_minor) >= 4:
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_87,code=sm_87')
            if int(bare_metal_minor) >= 8:
                cc_flag.append('-gencode')
                cc_flag.append('arch=compute_89,code=sm_89')
        if int(bare_metal_major) >= 12:
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_90,code=sm_90')

    build_dir = f"{os.path.expanduser('~')}/.cache/tgnn/build"
    if force_compile:
        shutil.rmtree(build_dir)

    os.makedirs(build_dir, exist_ok=True)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name,
                                   sources,
                                   extra_cuda_flags,
                                   extra_include_paths):
        extra_cuda_cflags = ['-O3',
                             '-gencode', 'arch=compute_70,code=sm_70',
                             '--use_fast_math'] + extra_cuda_flags + cc_flag
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=build_dir,
            extra_cflags=['-O3', ],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            verbose=verbose
        )

    current_dir = pathlib.Path(__file__).parent.absolute()
    if torch.version.hip is not None:
        extra_include_paths = [os.path.abspath(current_dir)]
    else:
        extra_include_paths = []

    if compile_rotary:
        extra_cuda_flags = [
            '--expt-extended-lambda']
        sources = [current_dir / 'rotary/rotary.cpp',
                   current_dir / 'rotary/rotary_cuda.cu']
        _cpp_extention_load_helper(
            "rotary_emb", sources, extra_cuda_flags, extra_include_paths)
        warnings.warn("using fused rotary embedding")
