#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#define CHECK_CUDNN(status) AT_CUDNN_CHECK(status, " at " __FILE__ ":", __LINE__)

struct cudnn_tensor_descriptor_deleter_t {
    void operator()(cudnnTensorDescriptor_t desc) { CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc)); }
};

using unique_cudnn_tensor_descriptor_t = std::unique_ptr<cudnnTensorStruct, cudnn_tensor_descriptor_deleter_t>;

inline unique_cudnn_tensor_descriptor_t make_unique_tensor_descriptor(torch::Tensor tensor) {
    std::vector<int> dims(tensor.sizes().begin(), tensor.sizes().end());
    std::vector<int> strides(tensor.strides().begin(), tensor.strides().end());
    cudnnTensorDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, tensor.ndimension(), dims.data(), strides.data()));
    return unique_cudnn_tensor_descriptor_t(desc);
}

struct cudnn_filter_descriptor_deleter_t {
    void operator()(cudnnFilterDescriptor_t desc) { CHECK_CUDNN(cudnnDestroyFilterDescriptor(desc)); }
};

using unique_cudnn_filter_descriptor_t = std::unique_ptr<cudnnFilterStruct, cudnn_filter_descriptor_deleter_t>;

inline unique_cudnn_filter_descriptor_t make_unique_filter_descriptor(torch::Tensor weight) {
    std::vector<int> dims(weight.sizes().begin(), weight.sizes().end());
    std::vector<int> strides(weight.strides().begin(), weight.strides().end());
    cudnnFilterDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&desc));
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dims.size(), dims.data()));
    return unique_cudnn_filter_descriptor_t(desc);
}

struct cudnn_convolution_descriptor_deleter_t {
    void operator()(cudnnConvolutionDescriptor_t desc) { CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(desc)); }
};

using unique_cudnn_convolution_descriptor_t =
    std::unique_ptr<cudnnConvolutionStruct, cudnn_convolution_descriptor_deleter_t>;

inline unique_cudnn_convolution_descriptor_t make_unique_convolution_descriptor(const std::vector<int> &stride,
                                                                                const std::vector<int> &padding,
                                                                                const std::vector<int> &dilation) {
    cudnnConvolutionDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&desc));
    CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(desc, padding.size(), padding.data(), stride.data(), dilation.data(),
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    return unique_cudnn_convolution_descriptor_t(desc);
}

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, std::optional<torch::Tensor> bias,
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

    const int algo_count = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    cudnnConvolutionFwdAlgoPerf_t perfs[algo_count];
    int returned_algo_count;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, weight_desc, conv_desc, output_desc,
                                                       algo_count, &returned_algo_count, perfs));
    TORCH_CHECK_GT(returned_algo_count, 0);
    cudnnConvolutionFwdAlgo_t algo = perfs[0].algo;

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

torch::Tensor conv_forward(torch::Tensor input, torch::Tensor weight, std::optional<torch::Tensor> bias,
                           const std::vector<int> &stride, const std::vector<int> &padding,
                           const std::vector<int> &dilation) {
    if (input.ndimension() == 3) {
        torch::Tensor output = conv_forward(input.unsqueeze(-2), weight.unsqueeze(-2), bias, {1, stride.at(0)},
                                            {0, padding.at(0)}, {1, dilation.at(0)});
        return output.squeeze(-2);
    }

    if (input.ndimension() == 4) {
        return conv2d_forward(input, weight, bias, stride, padding, dilation);
    }

    TORCH_CHECK(input.ndimension() >= 4);
    cudnnHandle_t handle = at::native::getCudnnHandle();

    auto conv_desc = make_unique_convolution_descriptor(stride, padding, dilation);

    auto input_desc = make_unique_tensor_descriptor(input);
    auto weight_desc = make_unique_filter_descriptor(weight);

    std::vector<int> output_dims(input.ndimension());
    CHECK_CUDNN(cudnnGetConvolutionNdForwardOutputDim(conv_desc.get(), input_desc.get(), weight_desc.get(),
                                                      output_dims.size(), output_dims.data()));
    TORCH_CHECK(output_dims.at(0) == input.size(0) && output_dims.at(1) == weight.size(0));

    torch::Tensor output = torch::empty(std::vector<long>(output_dims.begin(), output_dims.end()), input.options());
    auto output_desc = make_unique_tensor_descriptor(output);

    const float *input_ptr = input.const_data_ptr<float>();
    const float *weight_ptr = weight.const_data_ptr<float>();
    float *output_ptr = output.mutable_data_ptr<float>();

    const int algo_count = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    cudnnConvolutionFwdAlgoPerf_t perfs[algo_count];
    int returned_algo_count;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc.get(), weight_desc.get(), conv_desc.get(),
                                                       output_desc.get(), algo_count, &returned_algo_count, perfs));
    TORCH_CHECK_GT(returned_algo_count, 0);
    cudnnConvolutionFwdAlgo_t algo = perfs[0].algo;

    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc.get(), weight_desc.get(), conv_desc.get(),
                                                        output_desc.get(), algo, &workspace_size));

    auto workspace = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);

    const float alpha = 1.f;
    const float beta = 0.f;
    if (bias) {
        TORCH_CHECK(bias->ndimension() == 1 && bias->numel() == weight.size(0));
        std::vector<long> bias_dims(output.ndimension(), 1);
        bias_dims.at(1) = bias->numel();
        auto bias_desc = make_unique_tensor_descriptor(bias->view(bias_dims));

        const float *bias_ptr = bias->const_data_ptr<float>();

        cudnnActivationDescriptor_t act_desc;
        CHECK_CUDNN(cudnnCreateActivationDescriptor(&act_desc));
        CHECK_CUDNN(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0.0));

        CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
            handle, &alpha, input_desc.get(), input_ptr, weight_desc.get(), weight_ptr, conv_desc.get(), algo,
            workspace.get(), workspace_size, &beta, output_desc.get(), output_ptr, bias_desc.get(), bias_ptr, act_desc,
            output_desc.get(), output_ptr));

        CHECK_CUDNN(cudnnDestroyActivationDescriptor(act_desc));
    } else {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, input_desc.get(), input_ptr, weight_desc.get(), weight_ptr,
                                            conv_desc.get(), algo, workspace.get(), workspace_size, &beta,
                                            output_desc.get(), output_ptr));
    }

    return output;
}

torch::Tensor conv_backward_input(torch::Tensor grad_output, c10::IntArrayRef input_dims, torch::Tensor weight,
                                  const std::vector<int> &stride, const std::vector<int> &padding,
                                  const std::vector<int> &dilation) {
    if (grad_output.ndimension() == 3) {
        std::vector<long> unsqz_input_dims(input_dims.begin(), input_dims.end());
        unsqz_input_dims.insert(unsqz_input_dims.end() - 1, 1);
        torch::Tensor grad_input =
            conv_backward_input(grad_output.unsqueeze(-2), unsqz_input_dims, weight.unsqueeze(-2), {1, stride.at(0)},
                                {0, padding.at(0)}, {1, dilation.at(0)});
        return grad_input.squeeze(-2);
    }

    torch::Tensor grad_input = torch::empty(input_dims, weight.options());

    cudnnHandle_t handle = at::native::getCudnnHandle();

    auto conv_desc = make_unique_convolution_descriptor(stride, padding, dilation);

    auto grad_output_desc = make_unique_tensor_descriptor(grad_output);
    auto weight_desc = make_unique_filter_descriptor(weight);
    auto grad_input_desc = make_unique_tensor_descriptor(grad_input);

    const int algo_count = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    cudnnConvolutionBwdDataAlgoPerf_t perfs[algo_count];
    int returned_algo_count;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, weight_desc.get(), grad_output_desc.get(),
                                                            conv_desc.get(), grad_input_desc.get(), algo_count,
                                                            &returned_algo_count, perfs));
    TORCH_CHECK_GT(returned_algo_count, 0);
    cudnnConvolutionBwdDataAlgo_t algo = perfs[0].algo;

    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, weight_desc.get(), grad_output_desc.get(),
                                                             conv_desc.get(), grad_input_desc.get(), algo,
                                                             &workspace_size));
    auto workspace = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(handle, &alpha, weight_desc.get(), weight.const_data_ptr<float>(),
                                             grad_output_desc.get(), grad_output.const_data_ptr<float>(),
                                             conv_desc.get(), algo, workspace.get(), workspace_size, &beta,
                                             grad_input_desc.get(), grad_input.mutable_data_ptr<float>()));

    return grad_input;
}

torch::Tensor conv_backward_weight(torch::Tensor grad_output, torch::Tensor input, c10::IntArrayRef weight_dims,
                                   const std::vector<int> &stride, const std::vector<int> &padding,
                                   const std::vector<int> &dilation) {
    torch::Tensor grad_weight = torch::empty(weight_dims, input.options());

    cudnnHandle_t handle = at::native::getCudnnHandle();

    auto conv_desc = make_unique_convolution_descriptor(stride, padding, dilation);

    auto grad_output_desc = make_unique_tensor_descriptor(grad_output);
    auto input_desc = make_unique_tensor_descriptor(input);
    auto grad_weight_desc = make_unique_filter_descriptor(grad_weight);

    const int algo_count = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    cudnnConvolutionBwdFilterAlgoPerf_t perfs[algo_count];
    int returned_algo_count;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, input_desc.get(), grad_output_desc.get(),
                                                              conv_desc.get(), grad_weight_desc.get(), algo_count,
                                                              &returned_algo_count, perfs));
    TORCH_CHECK_GT(returned_algo_count, 0);
    cudnnConvolutionBwdFilterAlgo_t algo = perfs[0].algo;

    size_t workspace_size;
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc.get(), grad_output_desc.get(),
                                                               conv_desc.get(), grad_weight_desc.get(), algo,
                                                               &workspace_size));
    auto workspace = c10::cuda::CUDACachingAllocator::get()->allocate(workspace_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(handle, &alpha, input_desc.get(), input.const_data_ptr<float>(),
                                               grad_output_desc.get(), grad_output.const_data_ptr<float>(),
                                               conv_desc.get(), algo, workspace.get(), workspace_size, &beta,
                                               grad_weight_desc.get(), grad_weight.mutable_data_ptr<float>()));

    return grad_weight;
}

torch::Tensor conv_backward_bias(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
                                 const std::vector<int> &stride, const std::vector<int> &padding,
                                 const std::vector<int> &dilation) {
    std::vector<long> bias_dims(weight.ndimension(), 1);
    bias_dims.at(1) = weight.size(0); // set channel
    torch::Tensor grad_bias = torch::empty(bias_dims, weight.options());

    cudnnHandle_t handle = at::native::getCudnnHandle();

    auto conv_desc = make_unique_convolution_descriptor(stride, padding, dilation);

    auto grad_output_desc = make_unique_tensor_descriptor(grad_output);
    auto weight_desc = make_unique_filter_descriptor(weight);
    auto grad_bias_desc = make_unique_tensor_descriptor(grad_bias);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardBias(handle, &alpha, grad_output_desc.get(),
                                             grad_output.const_data_ptr<float>(), &beta, grad_bias_desc.get(),
                                             grad_bias.mutable_data_ptr<float>()));

    return grad_bias.view({grad_bias.size(1)});
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
conv_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight, const std::vector<int> &stride,
              const std::vector<int> &padding, const std::vector<int> &dilation, std::array<bool, 3> output_mask) {
    if (input.ndimension() == 3) {
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

    torch::Tensor grad_input;
    if (output_mask[0]) {
        grad_input = conv_backward_input(grad_output, input.sizes(), weight, stride, padding, dilation);
    }

    torch::Tensor grad_weight;
    if (output_mask[1]) {
        grad_weight = conv_backward_weight(grad_output, input, weight.sizes(), stride, padding, dilation);
    }

    torch::Tensor grad_bias;
    if (output_mask[2]) {
        grad_bias = conv_backward_bias(grad_output, input, weight, stride, padding, dilation);
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

torch::Tensor conv_transpose_forward(torch::Tensor input, torch::Tensor weight, std::optional<torch::Tensor> bias,
                                     const std::vector<int> &stride, const std::vector<int> &padding,
                                     const std::vector<int> &output_padding, const std::vector<int> &dilation) {
    std::vector<long> output_dims{input.size(0), weight.size(1)};
    for (size_t i = 0; i < stride.size(); i++) {
        const long output_dim = (input.size(i + 2) - 1) * stride[i] + dilation[i] * (weight.size(i + 2) - 1) + 1 -
                                2 * padding[i] + output_padding[i];
        output_dims.emplace_back(output_dim);
    }

    torch::Tensor output = conv_backward_input(input, output_dims, weight, stride, padding, dilation);
    if (bias) {
        std::vector<long> bias_dims(output.ndimension(), 1);
        bias_dims.at(1) = bias->numel();
        output.add_(bias->view(bias_dims));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward, "convolution using cudnn");
    m.def("conv_backward", &conv_backward, "convolution backward using cudnn");
    m.def("conv_transpose_forward", &conv_transpose_forward, "conv_transpose_forward using cudnn");
}
