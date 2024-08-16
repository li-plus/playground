from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gemv_w3",
    version="0.0.1",
    description="Example of PyTorch cpp and CUDA extensions",
    ext_modules=[
        CUDAExtension(
            name="gemv_w3",
            sources=["csrc/gemv_w3_cuda.cu", "csrc/gemv_w3.cpp"],
            extra_compile_args={"cxx": ["-g", "-O3"], "nvcc": ["-O3",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_HALF2_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        "--use_fast_math",
                        "--ptxas-options=-v",
                        ]},
        ),
    ],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension},
)
