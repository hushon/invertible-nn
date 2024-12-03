import torch
from torch import nn
from torch.autograd import Function
from typing import Callable, Tuple, Any
from dataclasses import dataclass


@dataclass
class OutputHooks:
    pop_hook: Callable
    push_hook: Callable


class ScaledInvertibleCouplingLayer(Function):
    """
    y1 = alpha1 * x1 + F(x2)
    y2 = alpha2 * x2 + G(y1)

    When alpha is set to zero, y2 is equivalent to the non-invertible layer.
    y1 = F(x2)
    y2 = G(F(x2))
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, F: nn.Module, G: nn.Module, alpha1: float, alpha2: float, output_hooks: OutputHooks = None):

        ctx.F = F
        ctx.G = G
        ctx.alpha1 = alpha1
        ctx.alpha2 = alpha2
        if output_hooks is not None:
            output_hooks.pop_hook()
            ctx.push_hook = output_hooks.push_hook

        y1 = alpha1 * x1 + F(x2.to(torch.float32)).to(x1.dtype)
        y2 = alpha2 * x2 + G(y1.to(torch.float32)).to(x2.dtype)

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
        alpha1 = ctx.alpha1
        alpha2 = ctx.alpha2
        y1, y2 = ctx.output

        # Reconstruct x1, x2
        with torch.enable_grad():
            y1_ = y1.detach().requires_grad_(True)
            g_y1 = G(y1_.to(torch.float32))
        x2 = (y2 - g_y1.detach().to(y2.dtype)).div_(alpha2)

        with torch.enable_grad():
            x2_ = x2.detach().requires_grad_(True)
            f_x2 = F(x2_.to(torch.float32))
        x1 = (y1 - f_x2.detach().to(y2.dtype)).div_(alpha1)

        # Compute gradients
        g_y1.backward(dy2.to(torch.float32))
        dx1 = dy1 + y1_.grad.to(dy1.dtype)

        f_x2.backward(dx1.to(torch.float32))
        dx2 = dy2.mul_(alpha2) + x2_.grad.to(dy2.dtype)
        dx1 = dx1.mul_(alpha1)

        if hasattr(ctx, "push_hook"):
            ctx.push_hook((x1.detach(), x2.detach()))

        return dx1, dx2, None, None, None, None, None


class ScaledCouplingBlock(nn.Module):
    x1_dtype = torch.float64
    x2_dtype = torch.float64

    def __init__(self, F: nn.Module, G: nn.Module, alpha1: float, alpha2: float, invert_for_backward=True):
        super().__init__()
        self.F = F
        self.G = G
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.invert_for_backward = invert_for_backward

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, output_hooks: OutputHooks = None):
        x1 = x1.to(self.x1_dtype)
        x2 = x2.to(self.x2_dtype)

        if self.invert_for_backward:
            y1, y2, output_hooks = ScaledInvertibleCouplingLayer.apply(x1, x2, self.F, self.G, self.alpha1, self.alpha2, output_hooks)
        else:
            y1 = self.alpha1 * x1 + self.F(x2.to(torch.float32)).to(self.x1_dtype)
            y2 = self.alpha2 * x2 + self.G(y1.to(torch.float32)).to(self.x2_dtype)
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
        ScaledCouplingBlock(get_mlp(), get_mlp(), 0.1, 0.1).to(dtype=dtype, device=device) for _ in range(17)
    ]
    


    # @torch.autocast(device.type, dtype=torch.bfloat16)
    def forward_loss_fn(x):
        nonlocal init_x1, init_x2, init_hook
        x1, x2 = x.chunk(2, dim=-1)
        output_hooks = None
        for i, block in enumerate(block_list):
            x1, x2, output_hooks = block(x1, x2, output_hooks)
            if i == 0:
                init_x1 = x1
                init_x2 = x2
                init_hook = output_hooks
        output = torch.cat([x1, x2], dim=-1)
        loss = output.norm()
        return loss
    
    init_x1 = init_x2 = init_hook = None

    # compute grad with activation inversion
    [module.zero_grad() for module in block_list]
    forward_loss_fn(input).backward()
    grad1 = input.grad.clone()

    rec_x1, rec_x2 = init_hook.pop_hook()
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