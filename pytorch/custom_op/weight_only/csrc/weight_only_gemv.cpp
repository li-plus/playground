#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

inline __half *to_cuda_native_ptr(torch::Half *ptr) { return (__half *)ptr; }
inline const __half *to_cuda_native_ptr(const torch::Half *ptr) { return (const __half *)ptr; }
inline __nv_bfloat16 *to_cuda_native_ptr(torch::BFloat16 *ptr) { return (__nv_bfloat16 *)ptr; }
inline const __nv_bfloat16 *to_cuda_native_ptr(const torch::BFloat16 *ptr) { return (const __nv_bfloat16 *)ptr; }

template <int group_size>
void gemv_w8_cuda(const __half *input, const uint8_t *weight, const __half *scales, const __half *bias, __half *output,
                  int M, int N);

torch::Tensor gemv_w8(torch::Tensor input, torch::Tensor weight, torch::Tensor scales, torch::Tensor bias) {
    // input: [N], weight: [M, N], output: [M]
    const int M = weight.size(0);
    const int N = weight.size(-1);

    TORCH_INTERNAL_ASSERT(input.is_cuda() && weight.is_cuda() && scales.is_cuda() && bias.is_cuda());

    constexpr int group_size = 128;
    TORCH_CHECK(input.is_contiguous() && input.numel() == N);
    TORCH_CHECK(weight.is_contiguous() && weight.numel() == M * N && N % 16 == 0 && weight.dtype() == torch::kUInt8);
    TORCH_CHECK(scales.is_contiguous() && scales.numel() * group_size == weight.numel() &&
                scales.dtype() == input.dtype());
    TORCH_CHECK(bias.is_contiguous() && bias.ndimension() == 1 && bias.numel() == M && bias.dtype() == input.dtype());

    auto output_shape = input.sizes().vec();
    output_shape.back() = M;
    torch::Tensor output = torch::empty(output_shape, input.options());

    AT_DISPATCH_SWITCH(input.scalar_type(), "gemv_w8_cuda", AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                           gemv_w8_cuda<group_size>(to_cuda_native_ptr(input.const_data_ptr<scalar_t>()),
                                                    weight.const_data_ptr<uint8_t>(),
                                                    to_cuda_native_ptr(scales.const_data_ptr<scalar_t>()),
                                                    to_cuda_native_ptr(bias.const_data_ptr<scalar_t>()),
                                                    to_cuda_native_ptr(output.mutable_data_ptr<scalar_t>()), M, N);
                       }));

    return output;
}

// Registers _C as a Python extension module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

// Defines the operators
TORCH_LIBRARY(weight_only, m) { m.def("gemv_w8(Tensor input, Tensor weight, Tensor scales, Tensor bias) -> Tensor"); }

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(weight_only, CUDA, m) { m.impl("gemv_w8", &gemv_w8); }
