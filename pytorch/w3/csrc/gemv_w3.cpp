#include "torch/types.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

// Convert torch::Half and torch::BFloat16 to half and nv_bfloat16, respectively
template <typename torch_scalar_t>
struct to_cuda_scalar {
    using type = torch_scalar_t;
};

template <>
struct to_cuda_scalar<torch::Half> {
    using type = half;
};

template <>
struct to_cuda_scalar<torch::BFloat16> {
    using type = nv_bfloat16;
};

template <typename torch_scalar_t>
using to_cuda_scalar_t = typename to_cuda_scalar<torch_scalar_t>::type;

// template <typename scalar_t, int group_size = 64>
// void gemv_nf4_cuda(const scalar_t *input, const uint8_t *weight, const float *scales, const scalar_t *bias,
//                    scalar_t *output, int M, int N);

// torch::Tensor gemv_nf4(torch::Tensor input, torch::Tensor weight, torch::Tensor scales, torch::Tensor bias) {
//     // input: [N], weight: [M, N], output: [M]
//     constexpr int group_size = 64;
//     TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.numel() == input.size(-1) &&
//                 input.nbytes() % (4 * sizeof(float)) == 0);
//     TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.ndimension() == 2 &&
//                 weight.dtype() == torch::kUInt8 && weight.nbytes() % (4 * sizeof(float)) == 0);
//     TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.numel() * group_size == weight.numel() * 2 &&
//                 scales.dtype() == torch::kFloat32);
//     TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.ndimension() == 1 &&
//                 bias.nbytes() % (4 * sizeof(float)) == 0);
//     TORCH_CHECK(weight.size(0) == bias.size(0) && weight.size(-1) * 2 == input.size(-1));
//     TORCH_CHECK(input.dtype() == bias.dtype());

//     const int M = weight.size(0);
//     const int N = input.size(-1);

//     auto output_shape = input.sizes().vec();
//     output_shape.back() = M;
//     torch::Tensor output = torch::empty(output_shape, input.options());

//     AT_DISPATCH_SWITCH(input.scalar_type(), "gemv_nf4_cuda", AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
//                            using cuda_scalar_t = to_cuda_scalar_t<scalar_t>;
//                            gemv_nf4_cuda((const cuda_scalar_t *)input.const_data_ptr<scalar_t>(),
//                                          weight.const_data_ptr<uint8_t>(), scales.const_data_ptr<float>(),
//                                          (const cuda_scalar_t *)bias.const_data_ptr<scalar_t>(),
//                                          (cuda_scalar_t *)output.mutable_data_ptr<scalar_t>(), M, N);
//                        }));

//     return output;
// }

void gemv_i3_cuda(const __half *input, const uint8_t *weight, const __half *scales, const __half *bias, __half *output,
                  int M, int N);

torch::Tensor gemv_i3(torch::Tensor input, torch::Tensor weight, torch::Tensor scales, torch::Tensor bias) {
    // void gemv_i3(const __half *input, const uint8_t *weight, const __half *scales, const __half* bias, __half
    // *output, int M, int N) {
    const int M = bias.size(0);
    const int N = input.size(-1);

    constexpr int group_size = 128;
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.numel() == input.size(-1) &&
                input.nbytes() % (4 * sizeof(float)) == 0);
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.ndimension() == 1 && weight.numel() * 8 / 3 == M* N &&
                weight.dtype() == torch::kUInt8 && weight.nbytes() % (4 * sizeof(float)) == 0);
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.numel() * group_size == M * N &&
                scales.dtype() == input.dtype());
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.ndimension() == 1 &&
                bias.nbytes() % (4 * sizeof(float)) == 0 && bias.dtype() == input.dtype());

    auto output_shape = input.sizes().vec();
    output_shape.back() = M;
    torch::Tensor output = torch::empty(output_shape, input.options());

    AT_DISPATCH_SWITCH(input.scalar_type(), "gemv_i3_cuda", AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
                           using cuda_scalar_t = to_cuda_scalar_t<scalar_t>;
                           gemv_i3_cuda((const cuda_scalar_t *)input.const_data_ptr<scalar_t>(),
                                        weight.const_data_ptr<uint8_t>(),
                                        (const cuda_scalar_t *)scales.const_data_ptr<scalar_t>(),
                                        (const cuda_scalar_t *)bias.const_data_ptr<scalar_t>(),
                                        (cuda_scalar_t *)output.mutable_data_ptr<scalar_t>(), M, N);
                       }));

    return output;
    // gemv_i3_cuda(input, weight.const_data_ptr<>(), const __half *scales, const __half *bias, __half *output, int M,
    // int N)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("gemv_nf4", &gemv_nf4);
    m.def("gemv_i3", &gemv_i3);
}
