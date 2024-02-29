#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor rms_norm_cuda_forward(torch::Tensor input, torch::Tensor weight, float eps);

std::vector<torch::Tensor> rms_norm_cuda_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                                  float eps);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

torch::Tensor rms_norm_forward(torch::Tensor input, torch::Tensor weight, float eps) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    return rms_norm_cuda_forward(input, weight, eps);
}

std::vector<torch::Tensor> rms_norm_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                             float eps) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    return rms_norm_cuda_backward(grad_output, input, weight, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_forward, "RMSNorm forward (CUDA)");
    m.def("backward", &rms_norm_backward, "RMSNorm backward (CUDA)");
}
