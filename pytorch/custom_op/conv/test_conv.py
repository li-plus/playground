import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension
from torch.utils.benchmark import Timer

debug = False
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
    def forward(ctx, input, weight, bias, stride, padding, dilation):
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bias_requires_grad = bias.requires_grad if bias is not None else False
        return conv_ops.conv_forward(input, weight, bias, stride, padding, dilation)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        output_mask = (input.requires_grad, weight.requires_grad, ctx.bias_requires_grad)
        grad_input, grad_weight, grad_bias = conv_ops.conv_backward(
            grad_output, input, weight, ctx.stride, ctx.padding, ctx.dilation, output_mask
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None


class CustomConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ConvFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation)


class CustomConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ConvFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation)


class CustomConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ConvFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation)


class CustomConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return conv_ops.conv_transpose_forward(
            input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.dilation
        )


class CustomConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return conv_ops.conv_transpose_forward(
            input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.dilation
        )


class CustomConvTranspose3d(nn.ConvTranspose3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return conv_ops.conv_transpose_forward(
            input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.dilation
        )


@pytest.mark.parametrize(
    "cls_ref, cls_opt, input_shape, check_backward",
    [
        (nn.Conv1d, CustomConv1d, (2, 8, 128), True),
        (nn.Conv2d, CustomConv2d, (2, 8, 128, 128), True),
        (nn.Conv3d, CustomConv3d, (2, 8, 128, 128, 128), True),
        (nn.ConvTranspose1d, CustomConvTranspose1d, (2, 8, 128), False),
        (nn.ConvTranspose2d, CustomConvTranspose2d, (2, 8, 128, 128), False),
        (nn.ConvTranspose3d, CustomConvTranspose3d, (2, 8, 128, 128, 128), False),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_conv(cls_ref, cls_opt, input_shape, bias, check_backward):
    input_ref = torch.randn(input_shape, device="cuda", requires_grad=True)
    input_opt = input_ref.detach().requires_grad_()

    kwargs = {}
    if cls_ref in (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d):
        kwargs.update(output_padding=1)

    model_ref = cls_ref(8, 16, kernel_size=3, stride=2, padding=1, dilation=3, **kwargs, bias=bias, device="cuda")
    model_opt = cls_opt(8, 16, kernel_size=3, stride=2, padding=1, dilation=3, **kwargs, bias=bias, device="cuda")
    model_opt.load_state_dict(model_ref.state_dict())

    output_ref = model_ref(input_ref)
    output_opt = model_opt(input_opt)
    torch.testing.assert_close(output_opt, output_ref)

    fwd_ref = Timer("model_ref(input_ref)", globals=locals()).timeit(10).mean * 1e6
    fwd_opt = Timer("model_opt(input_opt)", globals=locals()).timeit(10).mean * 1e6
    print(f"{cls_ref.__name__} forward: ref {fwd_ref:.2f} us, opt {fwd_opt:.2f} us")

    if not check_backward:
        return

    grad_output = torch.randn_like(output_ref)
    output_ref.backward(grad_output, retain_graph=True)
    output_opt.backward(grad_output, retain_graph=True)

    torch.testing.assert_close(input_opt.grad, input_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(model_opt.weight.grad, model_ref.weight.grad, rtol=1e-3, atol=1e-3)
    if bias:
        torch.testing.assert_close(model_opt.bias.grad, model_ref.bias.grad, rtol=1e-3, atol=1e-3)

    bwd_ref = Timer("output_ref.backward(grad_output, retain_graph=True)", globals=locals()).timeit(10).mean * 1e6
    bwd_opt = Timer("output_opt.backward(grad_output, retain_graph=True)", globals=locals()).timeit(10).mean * 1e6
    print(f"{cls_ref.__name__} backward: ref {bwd_ref:.2f} us, opt {bwd_opt:.2f} us")
