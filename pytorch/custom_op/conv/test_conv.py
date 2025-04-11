import torch.utils.cpp_extension
import torch.nn as nn
import torch.nn.functional as F
import pytest

debug = True
extra_cflags = ["-Og", '-g'] if debug else ['-O3']
extra_cuda_cflags = ['-O0', '-lineinfo'] if debug else ['-O3']

conv_ops = torch.utils.cpp_extension.load(
    name="custom_cuda",
    sources=["conv_ops.cu"],
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
    build_directory='build/',
    verbose=True,
)

@pytest.mark.parametrize('bias', [False, True])
def test_conv2d(bias):
    input = torch.randn(2, 8, 128, 128, device='cuda')
    model = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, dilation=3, bias=bias, device='cuda')
    output_ref = F.conv2d(input, model.weight, model.bias, model.stride, model.padding, model.dilation)
    output_opt = conv_ops.conv2d(input, model.weight, model.bias, model.stride, model.padding, model.dilation)
    torch.testing.assert_close(output_opt, output_ref)


@pytest.mark.parametrize('bias', [False, True])
def test_conv1d(bias):
    input = torch.randn(2, 8, 128, device='cuda')
    model = nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1, dilation=3, bias=bias, device='cuda')
    output_ref = F.conv1d(input, model.weight, model.bias, model.stride, model.padding, model.dilation)
    output_opt = conv_ops.conv1d(input, model.weight, model.bias, model.stride, model.padding, model.dilation)
    torch.testing.assert_close(output_opt, output_ref)


@pytest.mark.parametrize('bias', [False, True])
def test_conv3d(bias):
    input = torch.randn(2, 8, 128, 128, 128, device='cuda')
    model = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1, dilation=3, bias=bias, device='cuda')
    output_ref = F.conv3d(input, model.weight, model.bias, model.stride, model.padding, model.dilation)
    output_opt = conv_ops.conv3d(input, model.weight, model.bias, model.stride, model.padding, model.dilation)
    torch.testing.assert_close(output_opt, output_ref)


if __name__ =='__main__':
    test_conv1d()
    test_conv2d()
    test_conv3d()
