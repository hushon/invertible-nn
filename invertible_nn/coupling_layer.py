import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

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
    def forward(ctx, X_1: torch.Tensor, X_2: torch.Tensor, F: nn.Module, G: nn.Module, output_hooks: OutputHooks = None):
        ctx.F = F
        ctx.G = G
        if output_hooks is not None:
            output_hooks.pop_hook()
            ctx.push_hook = output_hooks.push_hook

        Y_1 = X_1 + F(X_2)
        Y_2 = X_2 + G(Y_1)

        ctx.output = (Y_1.detach(), Y_2.detach())

        def pop_hook():
            output = ctx.output
            del ctx.output
            return output

        def push_hook(x):
            ctx.output = x
        output_hooks = OutputHooks(pop_hook, push_hook)

        return Y_1, Y_2, output_hooks

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY_1, dY_2, _):
        F = ctx.F
        G = ctx.G
        Y_1, Y_2 = ctx.output

        with torch.enable_grad():
            Y_1.requires_grad_(True)
            g_Y_1 = G(Y_1)
            g_Y_1.backward(dY_2)
        Y_1_grad = Y_1.grad
        Y_1 = Y_1.detach()
        g_Y_1 = g_Y_1.detach()

        with torch.no_grad():
            X_2 = Y_2 - g_Y_1
            dX_1 = dY_1 + Y_1_grad

        with torch.enable_grad():
            X_2.requires_grad_(True)
            f_X_2 = F(X_2)
            f_X_2.backward(dX_1)
        X_2_grad = X_2.grad
        X_2 = X_2.detach()
        f_X_2 = f_X_2.detach()

        with torch.no_grad():
            dX_2 = dY_2 + X_2_grad

        if hasattr(ctx, "push_hook"):
            X_1 = Y_1 - f_X_2
            ctx.push_hook((X_1.detach(), X_2.detach()))

        return dX_1, dX_2, None, None, None


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


def finite_diff_grad_check_couplingblock():
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
    breakpoint()

    for module in block_list:
        module.invert_for_backward = False
    forward_loss_fn(input).backward()
    grad2 = input.grad.clone()
    breakpoint()

    # assert torch.allclose(grad1, grad2, atol=1e-5, rtol=1e-5)

    # forward_loss_fn(input).backward()
    


    if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
        print("Gradient check passed!")


if __name__ == "__main__":
    finite_diff_grad_check_couplingblock()
