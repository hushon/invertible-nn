import torch
from torch import nn
from torch.autograd import Function
from typing import Callable, Tuple, Any
from dataclasses import dataclass


@dataclass
class OutputHooks:
    pop_hook: Callable
    push_hook: Callable


class ResidualLayer(Function):
    """
    y1 = x + F(x)
    y2 = y1 + G(y1)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, F: nn.Module, G: nn.Module, output_hooks: OutputHooks = None):

        ctx.F = F
        ctx.G = G
        if output_hooks is not None:
            output_hooks.pop_hook()
            ctx.push_hook = output_hooks.push_hook

        y1 = x + F(x.to(torch.float32)).to(x.dtype)
        y2 = y1 + G(y1.to(torch.float32)).to(y1.dtype)

        ctx.output = y2.detach()

        def pop_hook():
            output = ctx.output
            del ctx.output
            return output

        def push_hook(x):
            ctx.output = x
        output_hooks = OutputHooks(pop_hook, push_hook)

        return y2, output_hooks

    @staticmethod
    def fixed_point_iteration(
        F: Callable,
        y: torch.Tensor,
        max_iter: int,
    ) -> torch.Tensor:
        x = y
        for _ in range(max_iter):
            x = y - F(x)
        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy2, _):

        F = ctx.F
        G = ctx.G
        y2 = ctx.output
        max_iter = 10

        # Reconstruct y1
        with torch.enable_grad():
            # y1 = ResidualLayer.fixed_point_iteration(G, y2.to(torch.float32), max_iter=1)
            y1 = y2
            for _ in range(max_iter):
                y1 = y1.detach().requires_grad_(True)
                g_y1 = G(y1.to(torch.float32))
                y1_ = y1
                y1 = y2 - g_y1.detach().to(y2.dtype)
            g_y1.backward(dy2.to(torch.float32))
        dy1 = dy2 + y1_.grad.to(dy2.dtype)

        # Reconstruct x
        with torch.enable_grad():
            # x = ResidualLayer.fixed_point_iteration(F, y1.to(torch.float32), max_iter=1)
            x = y1
            for _ in range(max_iter):
                x = x.detach().requires_grad_(True)
                f_x = F(x.to(torch.float32))
                x_ = x
                x = y1 - f_x.detach().to(y1.dtype)
            f_x.backward(dy1.to(torch.float32))
        dx = dy1 + x_.grad.to(dy1.dtype)

        if hasattr(ctx, "push_hook"):
            ctx.push_hook(x.detach())

        return dx, None, None, None


class ResidualBlock(nn.Module):
    x_dtype = torch.float64

    def __init__(self, F: nn.Module, G: nn.Module, invert_for_backward=True):
        super().__init__()
        self.F = F
        self.G = G
        self.invert_for_backward = invert_for_backward

    def forward(self, x: torch.Tensor, output_hooks: OutputHooks = None):
        x = x.to(self.x_dtype)

        if self.invert_for_backward:
            y2, output_hooks = ResidualLayer.apply(x, self.F, self.G, output_hooks)
        else:
            y1 = x + self.F(x.to(torch.float32)).to(self.x_dtype)
            y2 = y1 + self.G(y1.to(torch.float32)).to(self.x_dtype)
            output_hooks = None

        return y2, output_hooks


def grad_check():
    device = torch.device("cuda")
    # device = torch.device("cpu")
    dtype = torch.float32

    input = torch.rand(1, 8, requires_grad=True, dtype=dtype, device=device)

    get_mlp = lambda: nn.Sequential(
        nn.LayerNorm(8),
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8)
    )

    block_list = [
        ResidualBlock(get_mlp(), get_mlp()).to(dtype=dtype, device=device) for _ in range(10)
    ]
    


    # @torch.autocast(device.type, dtype=torch.bfloat16)
    def forward_loss_fn(x):
        nonlocal init_x, init_hook
        output_hooks = None
        for i, block in enumerate(block_list):
            x, output_hooks = block(x, output_hooks)
            if i == 0:
                init_x = x
                init_hook = output_hooks
        output = x
        loss = output.norm()
        return loss
    
    init_x = init_hook = None

    # compute grad with activation inversion
    [module.zero_grad() for module in block_list]
    forward_loss_fn(input).backward()
    grad1 = input.grad.clone()

    rec_x = init_hook.pop_hook()
    breakpoint()

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


def check_inverse():
    # device = torch.device("cuda")
    device = torch.device("cpu")
    dtype = torch.float32

    input = torch.rand(1, 16, requires_grad=True, dtype=dtype, device=device)

    get_mlp = lambda: nn.Sequential(
        nn.LayerNorm(8),
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8)
    )

    block = ScaledCouplingBlock(get_mlp(), get_mlp(), 0.1, 0.3).to(dtype=dtype, device=device)

    output_hooks = None
    y1, y2, output_hooks = block(*input.chunk(2, dim=-1), output_hooks)
    output = torch.cat([y1, y2], dim=-1)
    output.norm().backward()

    grad1 = input.grad

    input.grad.zero_()
    block.invert_for_backward = False
    output_hooks = None
    y1, y2, output_hooks = block(*input.chunk(2, dim=-1), output_hooks)
    output = torch.cat([y1, y2], dim=-1)
    output.norm().backward()

    grad2 = input.grad


    if torch.allclose(grad1, grad2, atol=1e-5, rtol=1e-5):
        print("allclose Gradient check passed!")
    else:
        print("allclose Gradient check failed!")

    if torch.nn.functional.cosine_similarity(grad1.view(-1), grad2.view(-1), dim=0) > 0.9:
        print("cosine_similarity Gradient check passed!")
    else:
        print("cosine_similarity Gradient check failed!")
    breakpoint()



if __name__ == "__main__":
    grad_check()
    # check_inverse()