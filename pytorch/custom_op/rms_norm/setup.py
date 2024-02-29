from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="rms-norm-cuda-op",
    ext_modules=[
        CUDAExtension(
            name="rms_norm_cuda_op",
            sources=["rms_norm_cuda.cpp", "rms_norm_cuda_kernel.cu"],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O3"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
