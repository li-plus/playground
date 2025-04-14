import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension

debug = True
extra_cflags = ["-Og", "-g"] if debug else ["-O3"]
extra_cuda_cflags = ["-O0", "-lineinfo"] if debug else ["-O3"]

conv_ops = torch.utils.cpp_extension.load(
    name="conv_ops",
    sources=["conv_ops.cu"],
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    build_directory="build/",
    verbose=True,
)


class ConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, conv_dim: int):
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.conv_dim = conv_dim
        ctx.bias_requires_grad = bias.requires_grad if bias is not None else False
        conv_fn_map = {1: conv_ops.conv1d, 2: conv_ops.conv2d, 3: conv_ops.conv3d}
        return conv_fn_map[conv_dim](input, weight, bias, stride, padding, dilation)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        output_mask = (input.requires_grad, weight.requires_grad, ctx.bias_requires_grad)
        conv_backward_map = {1: conv_ops.conv1d_backward, 2: conv_ops.conv2d_backward, 3: conv_ops.conv3d_backward}
        grad_input, grad_weight, grad_bias = conv_backward_map[ctx.conv_dim](
            grad_output, input, weight, ctx.stride, ctx.padding, ctx.dilation, output_mask
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None


class CustomConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ConvFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, 1)


class CustomConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ConvFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, 2)


class CustomConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ConvFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, 3)


@pytest.mark.parametrize(
    "cls_ref, cls_opt, input_shape",
    [
        (nn.Conv1d, CustomConv1d, (2, 8, 128)),
        (nn.Conv2d, CustomConv2d, (2, 8, 128, 128)),
        (nn.Conv3d, CustomConv3d, (2, 8, 128, 128, 128)),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_conv(cls_ref, cls_opt, input_shape, bias):
    input_ref = torch.randn(input_shape, device="cuda", requires_grad=True)
    input_opt = input_ref.detach().requires_grad_()
    model_ref = cls_ref(8, 16, kernel_size=3, stride=2, padding=1, dilation=3, bias=bias, device="cuda")
    model_opt = cls_opt(8, 16, kernel_size=3, stride=2, padding=1, dilation=3, bias=bias, device="cuda")
    model_opt.load_state_dict(model_ref.state_dict())

    output_ref = model_ref(input_ref)
    output_opt = model_opt(input_opt)
    torch.testing.assert_close(output_opt, output_ref)

    grad_output = torch.randn_like(output_ref)
    output_ref.backward(grad_output)
    output_opt.backward(grad_output)

    torch.testing.assert_close(input_opt.grad, input_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(model_opt.weight.grad, model_ref.weight.grad, rtol=1e-3, atol=1e-3)
    if bias:
        torch.testing.assert_close(model_opt.bias.grad, model_ref.bias.grad, rtol=1e-3, atol=1e-3)
