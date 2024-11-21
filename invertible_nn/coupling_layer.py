import torch
from torch import nn
from torch.autograd import Function
from typing import Callable, Tuple, Any
from dataclasses import dataclass


@dataclass
class OutputHooks:
    pop_hook: Callable
    push_hook: Callable


class InvertibleCouplingLayer(Function):
    """
    Y_1 = X_1 + F(X_2)
    Y_2 = X_2 + G(Y_1)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, F: nn.Module, G: nn.Module, output_hooks: OutputHooks = None):
        ctx.F = F
        ctx.G = G
        if output_hooks is not None:
            output_hooks.pop_hook()
            ctx.push_hook = output_hooks.push_hook

        y1 = x1 + F(x2)
        y2 = x2 + G(y1)

        ctx.output = (y1.detach(), y2.detach())

        def pop_hook():
            output = ctx.output
            del ctx.output
            return output

        def push_hook(x):
            ctx.output = x
        output_hooks = OutputHooks(pop_hook, push_hook)

        return y1, y2, output_hooks

    # @staticmethod
    # @torch.autograd.function.once_differentiable
    # @torch.cuda.amp.custom_bwd
    # def backward(ctx, dy1, dy2, _):
    #     F = ctx.F
    #     G = ctx.G
    #     y1, y2 = ctx.output

    #     with torch.enable_grad():
    #         y1_ = y1.detach().requires_grad_(True)
    #         g_y1 = G(y1_)
    #         g_y1.backward(dy2)
    #         g_y1 = g_y1.detach()

    #     x2 = y2 - g_y1
    #     dx1 = dy1 + y1_.grad

    #     with torch.enable_grad():
    #         x2_ = x2.detach().requires_grad_(True)
    #         f_x2 = F(x2_)
    #         f_x2.backward(dx1)
    #         f_x2 = f_x2.detach()

    #     x1 = y1 - f_x2
    #     dx2 = dy2 + x2_.grad

    #     if hasattr(ctx, "push_hook"):
    #         ctx.push_hook((x1.detach(), x2.detach()))

    #     return dx1, dx2, None, None, None
    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy1, dy2, _):
        F = ctx.F
        G = ctx.G
        y1, y2 = ctx.output

        # Reconstruct x1, x2
        with torch.enable_grad():
            y1_ = y1.detach().requires_grad_(True)
            g_y1 = G(y1_)
        x2 = y2 - g_y1.detach()

        with torch.enable_grad():
            x2_ = x2.detach().requires_grad_(True)
            f_x2 = F(x2_)
        x1 = y1 - f_x2.detach()

        # Compute gradients
        g_y1.backward(dy2)
        dx1 = dy1 + y1_.grad

        f_x2.backward(dx1)
        dx2 = dy2 + x2_.grad

        if hasattr(ctx, "push_hook"):
            ctx.push_hook((x1.detach(), x2.detach()))

        return dx1, dx2, None, None, None


class CouplingBlock(nn.Module):
    def __init__(self, F: nn.Module, G: nn.Module, invert_for_backward=True):
        super().__init__()
        self.F = F
        self.G = G
        self.invert_for_backward = invert_for_backward

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, output_hooks: OutputHooks = None):
        if self.invert_for_backward:
            y1, y2, output_hooks = InvertibleCouplingLayer.apply(x1, x2, self.F, self.G, output_hooks)
        else:
            y1 = x1 + self.F(x2)
            y2 = x2 + self.G(y1)
            output_hooks = None
        return y1, y2, output_hooks


def grad_check():
    # device = torch.device("cuda")
    device = torch.device("cpu")
    dtype = torch.float64

    input = torch.rand(1, 16, requires_grad=True, dtype=dtype, device=device)

    get_mlp = lambda: nn.Sequential(
        nn.LayerNorm(8),
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8)
    )
    # model = nn.Sequential(*[CouplingBlock(get_mlp(), get_mlp()) for _ in range(10)])
    # model = CouplingBlock(get_mlp(), get_mlp())

    block_list = [
        CouplingBlock(get_mlp(), get_mlp()).to(dtype=dtype, device=device) for _ in range(10)
    ]
    
    # @torch.autocast("cuda", dtype=torch.bfloat16)
    # @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def forward_loss_fn(x):
        x1, x2 = x.chunk(2, dim=-1)
        output_hooks = None
        for block in block_list:
            x1, x2, output_hooks = block(x1, x2, output_hooks)
        output = torch.cat([x1, x2], dim=-1)
        loss = output.norm()
        return loss
    

    forward_loss_fn(input).backward()
    grad1 = input.grad.clone()
    input.grad.zero_()
    [module.zero_grad() for module in block_list]

    for module in block_list:
        module.invert_for_backward = False
    forward_loss_fn(input).backward()
    grad2 = input.grad.clone()

    if torch.allclose(grad1, grad2, atol=1e-5, rtol=1e-5):
        print("Gradient check passed!")

    if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
        print("Gradient check passed!")


if __name__ == "__main__":
    grad_check()
