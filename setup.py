"""Build the CUDA extension with a compatible installed PyTorch/CUDA toolchain.

PyTorch selects the visible GPU architecture unless TORCH_CUDA_ARCH_LIST is set.
For cross-compilation or a headless builder, set that variable explicitly to the
intended deployment GPU(s). This build does not modify installed dependencies.
"""
import glob
import os.path as osp
import sys

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = osp.dirname(osp.abspath(__file__))
csrc = osp.join(HERE, "csrc")
sources = sorted(osp.relpath(p, HERE) for pattern in ("*.cpp", "*.cu")
                 for p in glob.glob(osp.join(csrc, pattern)))

# Do not add -gencode/-arch: those flags suppress PyTorch's normal architecture
# selection, including TORCH_CUDA_ARCH_LIST. Do not override CC/CXX either.
nvcc_flags = ["-O3", "--restrict", "-std=c++17"]
if sys.platform == "win32":
    cxx_flags = ["/O2", "/std:c++17", "/EHsc"]
    nvcc_flags += ["-Xcompiler", "/EHsc"]
else:
    cxx_flags = ["-O3", "-std=c++17", "-fPIC"]
    nvcc_flags += ["-Xcompiler", "-fPIC"]

setup(
    packages=find_packages(include=["neurosim_cu_esim", "neurosim_cu_esim.*"]),
    ext_modules=[CUDAExtension(
        name="_neurosim_cu_esim_ext",
        sources=sources,
        include_dirs=[osp.join(csrc, "include")],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )],
    cmdclass={"build_ext": BuildExtension},
)
