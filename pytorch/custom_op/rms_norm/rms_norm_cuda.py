import rms_norm_cuda_op
import torch


class RMSNormCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.eps = eps
        output = rms_norm_cuda_op.forward(input, weight, eps)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight = rms_norm_cuda_op.backward(grad_output.contiguous(), input, weight, ctx.eps)
        return grad_input, grad_weight, None


rms_norm_cuda = RMSNormCuda.apply
