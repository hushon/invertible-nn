import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function

from typing import Callable, Tuple, Any
from dataclasses import dataclass


@dataclass
class OutputHooks:
    pop_hook: Callable
    push_hook: Callable


class ScaledInvertibleCouplingLayer(Function):
    """
    y1 = alpha * x1 + F(x2)
    y2 = alpha * x2 + G(y1)

    When alpha is set to zero, y2 is equivalent to the non-invertible layer.
    y1 = F(x2)
    y2 = G(F(x2))
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, F: nn.Module, G: nn.Module, alpha: float, output_hooks: OutputHooks = None):

        ctx.F = F
        ctx.G = G
        ctx.alpha = alpha
        if output_hooks is not None:
            output_hooks.pop_hook()
            ctx.push_hook = output_hooks.push_hook

        y1 = alpha * x1 + F(x2.to(torch.float32)).to(x1.dtype)
        y2 = alpha * x2 + G(y1.to(torch.float32)).to(x2.dtype)

        ctx.output = (y1.detach(), y2.detach())

        def pop_hook():
            output = ctx.output
            del ctx.output
            return output

        def push_hook(x):
            ctx.output = x
        output_hooks = OutputHooks(pop_hook, push_hook)

        return y1, y2, output_hooks

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy1, dy2, _):

        F = ctx.F
        G = ctx.G
        alpha = ctx.alpha
        y1, y2 = ctx.output

        # Reconstruct x1, x2
        with torch.enable_grad():
            y1_ = y1.detach().requires_grad_(True)
            g_y1 = G(y1_.to(torch.float32))
        x2 = (y2 - g_y1.detach().to(y2.dtype)).div_(alpha)

        with torch.enable_grad():
            x2_ = x2.detach().requires_grad_(True)
            f_x2 = F(x2_.to(torch.float32))
        x1 = (y1 - f_x2.detach().to(y2.dtype)).div_(alpha)

        # Compute gradients
        g_y1.backward(dy2.to(torch.float32))
        dx1 = dy1 + y1_.grad.to(dy1.dtype)

        f_x2.backward(dx1.to(torch.float32))
        dx2 = dy2.mul_(alpha) + x2_.grad.to(dy2.dtype)
        dx1 = dx1.mul_(alpha)

        if hasattr(ctx, "push_hook"):
            ctx.push_hook((x1.detach(), x2.detach()))

        return dx1, dx2, None, None, None, None


class ScaledCouplingBlock(nn.Module):
    x1_dtype = torch.float64
    x2_dtype = torch.float64

    def __init__(self, F: nn.Module, G: nn.Module, alpha: float, invert_for_backward=True):
        super().__init__()
        self.F = F
        self.G = G
        self.alpha = alpha
        self.invert_for_backward = invert_for_backward

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, output_hooks: OutputHooks = None):
        x1 = x1.to(self.x1_dtype)
        x2 = x2.to(self.x2_dtype)

        if self.invert_for_backward:
            y1, y2, output_hooks = ScaledInvertibleCouplingLayer.apply(x1, x2, self.F, self.G, self.alpha, output_hooks)
        else:
            y1 = self.alpha * x1 + self.F(x2.to(torch.float32)).to(x1.dtype)
            y2 = self.alpha * x2 + self.G(y1.to(torch.float32)).to(x2.dtype)
            output_hooks = None
        return y1, y2, output_hooks


def grad_check():
    device = torch.device("cuda")
    # device = torch.device("cpu")
    dtype = torch.float32

    input = torch.rand(1, 16, requires_grad=True, dtype=dtype, device=device)

    get_mlp = lambda: nn.Sequential(
        nn.LayerNorm(8),
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8)
    )

    block_list = [
        ScaledCouplingBlock(get_mlp(), get_mlp(), 0.1).to(dtype=dtype, device=device) for _ in range(10)
    ]
    

    # @torch.autocast(device.type, dtype=torch.bfloat16)
    def forward_loss_fn(x):
        x1, x2 = x.chunk(2, dim=-1)
        output_hooks = None
        for block in block_list:
            x1, x2, output_hooks = block(x1, x2, output_hooks)
        output = torch.cat([x1, x2], dim=-1)
        loss = output.norm()
        return loss
    

    # compute grad with activation inversion
    [module.zero_grad() for module in block_list]
    forward_loss_fn(input).backward()
    grad1 = input.grad.clone()


    # compute grad without activation inversion
    for module in block_list:
        module.invert_for_backward = False
    input.grad.zero_()
    [module.zero_grad() for module in block_list]
    forward_loss_fn(input).backward()
    grad2 = input.grad.clone()

    if torch.allclose(grad1, grad2, atol=1e-5, rtol=1e-5):
        print("allclose Gradient check passed!")
    else:
        print("allclose Gradient check failed!")

    if torch.nn.functional.cosine_similarity(grad1.view(-1), grad2.view(-1), dim=0) > 0.9:
        print("cosine_similarity Gradient check passed!")
    else:
        print("cosine_similarity Gradient check failed!")
    breakpoint()

    # if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
    #     print("autograd.gradcheck Gradient check passed!")


if __name__ == "__main__":
    grad_check()
