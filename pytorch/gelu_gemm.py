import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class GeLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GeLULinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        return F.linear(F.gelu(x, approximate="tanh"), w, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, w = ctx.saved_tensors
        x_act = F.gelu(x, approximate="tanh")  # [b, s, 4h]
        grad_x_act = grad_output @ w  # [b, s, 4h] @ [4h, h] -> [b, s, h]
        grad_w = grad_output.flatten(end_dim=-2).T @ x_act.flatten(end_dim=-2)  # [4h, b*s] @ [b*s, h] -> [4h, h]
        grad_b = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        grad_x = torch.ops.aten.gelu_backward(grad_x_act, x, approximate="tanh")
        return grad_x, grad_w, grad_b


class FusedGeLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)
        assert "gelu" in hidden_act, f"not implemented for {hidden_act} activation"

    def forward(self, x) -> torch.Tensor:
        return GeLULinearFunction.apply(self.fc1(x), self.fc2.weight, self.fc2.bias)


def test_gelu_linear_function():
    ref_x = torch.randn(3, 17, 4096, device="cuda", requires_grad=True)
    ref_w = torch.randn(1024, 4096, device="cuda", requires_grad=True)
    ref_b = torch.randn(1024, device="cuda", requires_grad=True)

    opt_x = ref_x.detach().clone().requires_grad_()
    opt_w = ref_w.detach().clone().requires_grad_()
    opt_b = ref_b.detach().clone().requires_grad_()

    ref_output = F.linear(F.gelu(ref_x, approximate="tanh"), ref_w, ref_b)
    opt_output = GeLULinearFunction.apply(opt_x, opt_w, opt_b)

    torch.testing.assert_close(opt_output, ref_output)

    grad_output = torch.randn_like(ref_output)

    ref_output.backward(grad_output)
    opt_output.backward(grad_output)

    torch.testing.assert_close(opt_x.grad, ref_x.grad)
    torch.testing.assert_close(opt_w.grad, ref_w.grad)
    torch.testing.assert_close(opt_b.grad, ref_b.grad)


def test_fused_gelu_mlp():
    ref_model = GeLUMLP(1024, 4096, "gelu_pytorch_tanh").cuda()
    opt_model = FusedGeLUMLP(1024, 4096, "gelu_pytorch_tanh").cuda()

    opt_model.load_state_dict(ref_model.state_dict())

    ref_x = torch.randn(3, 17, 1024, device="cuda", requires_grad=True)
    opt_x = ref_x.detach().clone().requires_grad_()

    ref_y = ref_model(ref_x)
    opt_y = opt_model(opt_x)

    torch.testing.assert_close(opt_y, ref_y)

    grad_output = torch.randn_like(ref_y)
    ref_y.backward(grad_output)
    opt_y.backward(grad_output)

    torch.testing.assert_close(opt_x.grad, ref_x.grad)


test_gelu_linear_function()
test_fused_gelu_mlp()
