#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#define CHECK_CUDNN(status) AT_CUDNN_CHECK(status, " at " __FILE__ ":", __LINE__)

torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight, std::optional<torch::Tensor> bias,
                     const std::vector<int> &stride, const std::vector<int> &padding,
                     const std::vector<int> &dilation) {
    TORCH_CHECK(stride.size() == 2);
    TORCH_CHECK(padding.size() == 2);
    TORCH_CHECK(dilation.size() == 2);

    cudnnHandle_t handle = at::native::getCudnnHandle();

    const int N = input.size(0);
    const int IC = input.size(1);
    const int IH = input.size(2);
    const int IW = input.size(3);

    const int OC = weight.size(0);
    TORCH_CHECK(IC == weight.size(1));
    const int KH = weight.size(2);
    const int KW = weight.size(3);

    int PH = padding.at(0), PW = padding.at(1);
    int SH = stride.at(0), SW = stride.at(1);
    int DH = dilation.at(0), DW = dilation.at(1);

    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(
        cudnnSetConvolution2dDescriptor(conv_desc, PH, PW, SH, SW, DH, DW, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, IC, IH, IW));

    cudnnFilterDescriptor_t weight_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, OC, IC, KH, KW));

    int ON, OC_COMP, OH, OW;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, weight_desc, &ON, &OC_COMP, &OH, &OW));
    TORCH_CHECK(ON == N && OC_COMP == OC);

    cudnnTensorDescriptor_t output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, OC, OH, OW));

    torch::Tensor output = torch::empty({N, OC, OH, OW}, input.options());

    const float *input_ptr = input.const_data_ptr<float>();
    const float *weight_ptr = weight.const_data_ptr<float>();
    float *output_ptr = output.mutable_data_ptr<float>();

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, weight_desc, conv_desc, output_desc, algo,
                                                        &workspace_size));

    auto workspace = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);

    const float alpha = 1.f;
    const float beta = 0.f;
    if (bias) {
        TORCH_CHECK(bias->ndimension() == 1 && bias->numel() == OC);
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, OC, 1, 1));
        const float *bias_ptr = bias->const_data_ptr<float>();

        cudnnActivationDescriptor_t act_desc;
        CHECK_CUDNN(cudnnCreateActivationDescriptor(&act_desc));
        CHECK_CUDNN(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            handle, &alpha, input_desc, input_ptr, weight_desc, weight_ptr, conv_desc, algo, workspace.get(),
            workspace_size, &beta, output_desc, output_ptr, bias_desc, bias_ptr, act_desc, output_desc, output_ptr));

        CHECK_CUDNN(cudnnDestroyActivationDescriptor(act_desc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    } else {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, input_desc, input_ptr, weight_desc, weight_ptr, conv_desc,
                                            algo, workspace.get(), workspace_size, &beta, output_desc, output_ptr));
    }

    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(weight_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));

    return output;
}

torch::Tensor convnd(torch::Tensor input, torch::Tensor weight, std::optional<torch::Tensor> bias,
                     const std::vector<int> &stride, const std::vector<int> &padding,
                     const std::vector<int> &dilation) {
    TORCH_CHECK(input.ndimension() >= 3);
    cudnnHandle_t handle = at::native::getCudnnHandle();

    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv_desc, stride.size(), padding.data(), stride.data(),
                                                dilation.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    std::vector<int> input_dims(input.sizes().begin(), input.sizes().end());
    std::vector<int> input_strides(input.strides().begin(), input.strides().end());
    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, input.ndimension(), input_dims.data(),
                                           input_strides.data()));

    std::vector<int> weight_dims(weight.sizes().begin(), weight.sizes().end());
    cudnnFilterDescriptor_t weight_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weight.ndimension(),
                                           weight_dims.data()));

    std::vector<int> output_dims(input.ndimension());

    CHECK_CUDNN(cudnnGetConvolutionNdForwardOutputDim(conv_desc, input_desc, weight_desc, output_dims.size(),
                                                      output_dims.data()));
    TORCH_CHECK(output_dims.at(0) == input_dims.at(0) && output_dims.at(1) == weight_dims.at(0));

    torch::Tensor output = torch::empty(std::vector<long>(output_dims.begin(), output_dims.end()), input.options());
    std::vector<int> output_strides(output.strides().begin(), output.strides().end());

    cudnnTensorDescriptor_t output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, output.ndimension(), output_dims.data(),
                                           output_strides.data()));

    const float *input_ptr = input.const_data_ptr<float>();
    const float *weight_ptr = weight.const_data_ptr<float>();
    float *output_ptr = output.mutable_data_ptr<float>();

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, weight_desc, conv_desc, output_desc, algo,
                                                        &workspace_size));

    auto workspace = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);

    const float alpha = 1.f;
    const float beta = 0.f;
    if (bias) {
        TORCH_CHECK(bias->ndimension() == 1 && bias->numel() == weight.size(0));
        std::vector<int> bias_dims(output.ndimension(), 1);
        bias_dims.at(1) = bias->numel();
        std::vector<int> bias_strides(output.ndimension(), bias->stride(0));
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, bias_dims.size(), bias_dims.data(),
                                               bias_strides.data()));
        const float *bias_ptr = bias->const_data_ptr<float>();

        cudnnActivationDescriptor_t act_desc;
        CHECK_CUDNN(cudnnCreateActivationDescriptor(&act_desc));
        CHECK_CUDNN(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            handle, &alpha, input_desc, input_ptr, weight_desc, weight_ptr, conv_desc, algo, workspace.get(),
            workspace_size, &beta, output_desc, output_ptr, bias_desc, bias_ptr, act_desc, output_desc, output_ptr));

        CHECK_CUDNN(cudnnDestroyActivationDescriptor(act_desc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    } else {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, input_desc, input_ptr, weight_desc, weight_ptr, conv_desc,
                                            algo, workspace.get(), workspace_size, &beta, output_desc, output_ptr));
    }

    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(weight_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));

    return output;
}

torch::Tensor conv1d(torch::Tensor input, torch::Tensor weight, std::optional<torch::Tensor> bias,
                     const std::vector<int> &stride, const std::vector<int> &padding,
                     const std::vector<int> &dilation) {
    torch::Tensor output = ::conv2d(input.unsqueeze(-2), weight.unsqueeze(-2), bias, {1, stride.at(0)},
                                    {0, padding.at(0)}, {1, dilation.at(0)});
    return output.squeeze(-2);
}

inline cudnnTensorDescriptor_t create_tensor_descriptor(torch::Tensor tensor) {
    std::vector<int> tensor_dims(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int> tensor_strides(tensor.strides().begin(), tensor.strides().end());
    cudnnTensorDescriptor_t tensor_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensor_desc));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(tensor_desc, CUDNN_DATA_FLOAT, tensor.ndimension(), tensor_dims.data(),
                                           tensor_strides.data()));
    return tensor_desc;
}

inline void destroy_tensor_descriptor(cudnnTensorDescriptor_t tensor_desc) {
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensor_desc));
}

inline cudnnFilterDescriptor_t create_filter_descriptor(torch::Tensor weight) {
    std::vector<int> weight_dims(weight.sizes().begin(), weight.sizes().end());
    std::vector<int> weight_strides(weight.strides().begin(), weight.strides().end());
    cudnnFilterDescriptor_t weight_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weight_dims.size(),
                                           weight_dims.data()));
    return weight_desc;
}

inline void destroy_filter_descriptor(cudnnFilterDescriptor_t filter_desc) {
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
}

inline cudnnConvolutionDescriptor_t create_convolution_descriptor(const std::vector<int> &stride,
                                                                  const std::vector<int> &padding,
                                                                  const std::vector<int> &dilation) {
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(conv_desc, padding.size(), padding.data(), stride.data(),
                                                dilation.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    return conv_desc;
}

inline void destroy_convolution_descriptor(cudnnConvolutionDescriptor_t conv_desc) {
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
}

torch::Tensor conv_backward_input(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                  const std::vector<int> &stride, const std::vector<int> &padding,
                                  const std::vector<int> &dilation) {
    torch::Tensor grad_input = torch::empty_like(input);

    cudnnHandle_t handle = at::native::getCudnnHandle();

    cudnnConvolutionDescriptor_t conv_desc = create_convolution_descriptor(stride, padding, dilation);

    cudnnTensorDescriptor_t grad_output_desc = create_tensor_descriptor(grad_output);
    cudnnFilterDescriptor_t weight_desc = create_filter_descriptor(weight);
    cudnnTensorDescriptor_t grad_input_desc = create_tensor_descriptor(grad_input);

    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, weight_desc, grad_output_desc, conv_desc,
                                                             grad_input_desc, algo, &workspace_size));
    auto workspace = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(handle, &alpha, weight_desc, weight.const_data_ptr<float>(),
                                             grad_output_desc, grad_output.const_data_ptr<float>(), conv_desc, algo,
                                             workspace.get(), workspace_size, &beta, grad_input_desc,
                                             grad_input.mutable_data_ptr<float>()));

    destroy_convolution_descriptor(conv_desc);

    destroy_tensor_descriptor(grad_output_desc);
    destroy_filter_descriptor(weight_desc);
    destroy_tensor_descriptor(grad_input_desc);

    return grad_input;
}

torch::Tensor conv_backward_weight(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                   const std::vector<int> &stride, const std::vector<int> &padding,
                                   const std::vector<int> &dilation) {
    torch::Tensor grad_weight = torch::empty_like(weight);

    cudnnHandle_t handle = at::native::getCudnnHandle();

    cudnnConvolutionDescriptor_t conv_desc = create_convolution_descriptor(stride, padding, dilation);

    cudnnTensorDescriptor_t grad_output_desc = create_tensor_descriptor(grad_output);
    cudnnTensorDescriptor_t input_desc = create_tensor_descriptor(input);
    cudnnFilterDescriptor_t grad_weight_desc = create_filter_descriptor(grad_weight);

    cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, grad_output_desc, conv_desc,
                                                               grad_weight_desc, algo, &workspace_size));
    auto workspace = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUDNN(cudnnConvolutionBackwardFilter(handle, &alpha, input_desc, input.const_data_ptr<float>(),
                                               grad_output_desc, grad_output.const_data_ptr<float>(), conv_desc, algo,
                                               workspace.get(), workspace_size, &beta, grad_weight_desc,
                                               grad_weight.mutable_data_ptr<float>()));

    destroy_convolution_descriptor(conv_desc);

    destroy_tensor_descriptor(grad_output_desc);
    destroy_tensor_descriptor(input_desc);
    destroy_filter_descriptor(grad_weight_desc);

    return grad_weight;
}

torch::Tensor conv_backward_bias(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                 const std::vector<int> &stride, const std::vector<int> &padding,
                                 const std::vector<int> &dilation) {
    std::vector<long> bias_dims(weight.ndimension(), 1);
    bias_dims.at(1) = weight.size(0); // set channel
    torch::Tensor grad_bias = torch::empty(bias_dims, weight.options());

    cudnnHandle_t handle = at::native::getCudnnHandle();

    cudnnConvolutionDescriptor_t conv_desc = create_convolution_descriptor(stride, padding, dilation);

    cudnnTensorDescriptor_t grad_output_desc = create_tensor_descriptor(grad_output);
    cudnnFilterDescriptor_t weight_desc = create_filter_descriptor(weight);
    cudnnTensorDescriptor_t grad_bias_desc = create_tensor_descriptor(grad_bias);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardBias(handle, &alpha, grad_output_desc, grad_output.const_data_ptr<float>(),
                                             &beta, grad_bias_desc, grad_bias.mutable_data_ptr<float>()));

    destroy_convolution_descriptor(conv_desc);

    destroy_tensor_descriptor(grad_output_desc);
    destroy_filter_descriptor(weight_desc);

    return grad_bias.view({grad_bias.size(1)});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
conv_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, const std::vector<int> &stride,
              const std::vector<int> &padding, const std::vector<int> &dilation, std::array<bool, 3> output_mask) {
    torch::Tensor grad_input;
    if (output_mask[0]) {
        grad_input = conv_backward_input(grad_output, input, weight, stride, padding, dilation);
    }

    torch::Tensor grad_weight;
    if (output_mask[1]) {
        grad_weight = conv_backward_weight(grad_output, input, weight, stride, padding, dilation);
    }

    torch::Tensor grad_bias;
    if (output_mask[2]) {
        grad_bias = conv_backward_bias(grad_output, input, weight, stride, padding, dilation);
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
conv1d_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, const std::vector<int> &stride,
                const std::vector<int> &padding, const std::vector<int> &dilation, std::array<bool, 3> output_mask) {
    auto [grad_input, grad_weight, grad_bias] =
        conv_backward(grad_output.unsqueeze(-2), input.unsqueeze(-2), weight.unsqueeze(-2), {1, stride.at(0)},
                      {0, padding.at(0)}, {1, dilation.at(0)}, output_mask);

    if (output_mask[0]) {
        grad_input.squeeze_(-2);
    }
    if (output_mask[1]) {
        grad_weight.squeeze_(-2);
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &convnd, "conv3d using cudnn");
    m.def("conv2d", &conv2d, "conv2d using cudnn");
    m.def("conv1d", &conv1d, "conv1d using cudnn");
    m.def("conv1d_backward", &conv1d_backward, "conv1d_backward using cudnn");
    m.def("conv2d_backward", &conv_backward, "conv2d_backward using cudnn");
    m.def("conv3d_backward", &conv_backward, "conv3d_backward using cudnn");
}
