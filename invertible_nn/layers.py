import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

from typing import Callable, Tuple, Any


class InvertibleCouplingLayer(Function):

    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, F: Callable, G: Callable) -> torch.Tensor:
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        forward pass equations:
        Y_1 = X_1 + F(X_2), F = Attention
        Y_2 = X_2 + G(Y_1), G = MLP
        """
        breakpoint()
        ctx.F = F
        ctx.G = G
        if hasattr(x, "release_saved_output"):  ## invertible layer 의 출력인지 확인
            x.release_saved_output()  # ctx.saved_tensors 에 있는 레퍼런스 삭제
            ctx.save_output = x.save_output
        ctx.requires_output = hasattr(x, "release_saved_output")  # 다음 레이어야 output 을 저장해야하는지 확인

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)
        del x


        Y_1 = X_1 + F(X_2)

        # free memory since X_1 is now not needed
        del X_1

        Y_2 = X_2 + G(Y_1)

        # free memory since X_2 is now not needed
        del X_2
        
        output = torch.cat([Y_1, Y_2], dim=-1)
        del Y_1, Y_2
        # output.save_for_backward = ctx.save_for_backward  ## 이러면 forward 할때 output이 free 될수 있나?
        ## 일단 저장하고, 뒤에 오는게 InvertibleCouplingLayer 타입이면 직접 해제하는 식으로 처리?
        ## 근데 기본적으로 해제될텐데? explicit 하게 저장하도록 하는게 맞는듯
        ## forward 할때 default 는 해제임.
        ## 다음 레이어가 InvertibleCouplingLayer 타입이 아닌지 체크하고, 아니라면 저장하도록 하는게 맞는듯
        ## invertible layer 가 아니면 save_for_backward 호출하도록 하는 방법..?
        # 입력이 invertible 레이어에서 온 것인지 확인
        ctx.output = output.detach()

        def release_saved_output():
            del ctx.output
            # delattr(ctx, 'output')
        def save_output(x):
            ctx.output = x
        output.release_saved_output = release_saved_output
        output.save_output = save_output
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        y = ctx.output
        F, G = ctx.F, ctx.G

        # obtaining gradients dX_1 and dX_2 from the concatenated input
        Y_1, Y_2 = torch.chunk(y, 2, dim=-1)
        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad_(True)

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1 = G(Y_1)
            g_Y_1.backward(dY_2)
        Y_1_grad = Y_1.grad
        Y_1 = Y_1.detach()
        g_Y_1 = g_Y_1.detach()

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dX_1 = dY_1 + Y_1_grad

            # free memory since Y_1.grad is now not needed
            del Y_1_grad

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad_(True)

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2 = F(X_2)
            f_X_2.backward(dX_1)
        X_2_grad = X_2.grad
        X_2 = X_2.detach()
        f_X_2 = f_X_2.detach()

        with torch.no_grad():
            dX_2 = dY_2 + X_2_grad
            del X_2_grad

        dx = torch.cat([dX_1, dX_2], dim=-1)

        ## 여기서 다음 backward pass 에 X_1, X_2 를 전달해줘야함
        if ctx.requires_output:
            X_1 = Y_1 - f_X_2
            del f_X_2
            x = torch.cat([X_1, X_2], dim=-1)
            ctx.save_output(x.detach())
        return dx, None, None, None


class CouplingBlock(nn.Module):
    """
    F 랑 G 는 임의의 모듈
    F랑 G를 coupling 구조에 끼워넣음.
    backward pass 할때는 뒷쪽 블락에서 보내준 activation 값을 이용해 중간값 재계산
    Y_1 = X_1 + F(X_2)
    Y_2 = X_2 + G(Y_1)
    """
    def __init__(self, F: nn.Module, G: nn.Module, invert_when_backward=True):
        super().__init__()
        self.F = F
        self.G = G
        self.invert_when_backward = invert_when_backward

    def forward(self, x):
        if self.invert_when_backward:
            return InvertibleCouplingLayer.apply(x, self.F, self.G)
        else:
            X_1, X_2 = torch.chunk(x, 2, dim=-1)
            Y_1 = X_1 + self.F(X_2)
            Y_2 = X_2 + self.G(Y_1)
            return torch.cat([Y_1, Y_2], dim=-1)


class InvertibleCouplingLayerV2(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X_1: torch.Tensor, X_2: torch.Tensor, F: nn.Module, G: nn.Module) -> torch.Tensor:
        """
        Y_1 = X_1 + F(X_2)
        Y_2 = X_2 + G(Y_1)
        """
        ctx.F = F
        ctx.G = G
        if hasattr(X_1, "release_saved_output"):
            X_1.release_saved_output()  # ctx.saved_tensors 에 있는 레퍼런스 삭제
            ctx.save_output = X_1.save_output

        Y_1 = X_1 + F(X_2)
        Y_2 = X_2 + G(Y_1)

        ctx.output = (Y_1.detach(), Y_2.detach())

        def release_saved_output():
            del ctx.output
        def save_output(x):
            ctx.output = x
        Y_1.release_saved_output = release_saved_output
        Y_1.save_output = save_output

        output = (Y_1, Y_2)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy1: torch.Tensor, dy2: torch.Tensor) -> torch.Tensor:
        F = ctx.F
        G = ctx.G
        Y_1, Y_2 = ctx.output
        dY_1, dY_2 = dy1, dy2

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

        if ctx.requires_output:  ## 여기서 다음 backward pass 에 X_1, X_2 를 전달해줘야함
            X_1 = Y_1 - f_X_2
            ctx.save_output((X_1.detach(), X_2.detach()))
        return dX_1, dX_2, None, None, None


class CouplingBlockV2(nn.Module):
    def __init__(self, F: nn.Module, G: nn.Module, invert_when_backward=True, pass_cache_to_prev_layer=False):
        super().__init__()
        self.F = F
        self.G = G
        self.invert_when_backward = invert_when_backward
        self.pass_cache_to_prev_layer = pass_cache_to_prev_layer

    def forward(self, x: torch.Tensor, ) -> torch.Tensor:
        if not self.invert_when_backward:
            X_1, X_2 = torch.chunk(x, 2, dim=-1)
            Y_1 = X_1 + self.F(X_2)
            Y_2 = X_2 + self.G(Y_1)
            return torch.cat([Y_1, Y_2], dim=-1)

        if self.pass_cache_to_prev_layer:

            # 입력 grad_fn 의 saved_tensor 를 free 하고 backward 때 다시 받도록 하는 hook 을 등록
            # 1. 이전 레이어의 grad_fn 에 있는 saved_tensor 를 제거
            # 2. autograd.Function 에 saved_tensor 를 등록하는 훅 전달

            def pack_hook(saved_tensor):
                breakpoint()  # substitute saved_tensor with None
                return None

            def unpack_hook(saved_tensor):
                # saved_tensor is expected to be None
                return pop_cache()  # pop from cache

            def push_cache(tensor):
                # send saved_tensor to the unpack_hook
                NotImplemented

            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                return InvertibleCouplingLayerV2.apply(x, self.F, self.G, push_cache)
        
        return InvertibleCouplingLayerV2.apply(x, self.F, self.G)



class InvertibleResidualLayer(Function):

    @staticmethod
    def fixed_point_iteration(
        F: Callable,
        y: torch.Tensor,
        max_iter: int,
        atol: float = 1e-5,
        verbose: bool = True
    ) -> torch.Tensor:
        x = y
        for _ in range(max_iter):
            x_prev = x
            x = y - F(x_prev)
            if torch.allclose(x, x_prev, atol=atol):
                break
        else:  # when loop did not break
            if verbose:
                print("Fixed point iteration did not converge.")
        return x

    @staticmethod
    def anderson_acceleration(F, y, max_iter, atol=1e-5, m=5):
        pass

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, F: Callable, max_iter: int, use_anderson_acceleration: bool) -> torch.Tensor:
        """
        forward pass equations:
        y = x + F(x)
        F(x) must be 1-Lipschitz
        """
        ctx.F = F
        if hasattr(x, "release_saved_output"):  ## invertible layer 의 출력인지 확인
            x.release_saved_output()  # ctx.saved_tensors 에 있는 레퍼런스 삭제
            ctx.save_output = x.save_output
        ctx.requires_output = hasattr(x, "release_saved_output")  # 다음 레이어야 output 을 저장해야하는지 확인

        output = x + F(x)

        ctx.output = output.detach()
        ctx.max_iter = max_iter
        ctx.use_anderson_acceleration = use_anderson_acceleration

        def release_saved_output():
            del ctx.output
            # delattr(ctx, 'output')
        def save_output(x):
            ctx.output = x
        output.release_saved_output = release_saved_output
        output.save_output = save_output

        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        y = ctx.output
        F = ctx.F
        max_iter = ctx.max_iter
        use_anderson_acceleration = ctx.use_anderson_acceleration

        # reconstruct x from y
        if use_anderson_acceleration:
            x = InvertibleResidualLayer.anderson_acceleration(F, y, max_iter)
        else:
            x = InvertibleResidualLayer.fixed_point_iteration(F, y, max_iter)

        with torch.enable_grad():
            x.requires_grad_(True)
            F(x).backward(dy)
            dx = x.grad + dy
            x = x.detach()

        if ctx.requires_output:
            ctx.save_output(x.detach())

        return dx, None, None, None


class ResidualBlock(nn.Module):
    def __init__(self, F: nn.Module):
        super().__init__()
        self.F = F
        self.F.apply(self.apply_spectral_normalization)

    @staticmethod
    def apply_spectral_normalization(module):
        if hasattr(module, "weight"):
            torch.nn.utils.parametrizations.spectral_norm(module, n_power_iterations=50)

    def forward(self, x):
        return InvertibleResidualLayer.apply(x, self.F, 100, False)


class InvertibleMomentumResidualLayer(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x: torch.Tensor, F: Callable, G: Callable) -> torch.Tensor:
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass pass logic.
        forward pass equations:
        Y_1 = X_1 + F(X_2), F = Attention
        Y_2 = X_2 + G(Y_1), G = MLP
        """
        ctx.F = F
        ctx.G = G
        if hasattr(x, "release_saved_output"):  ## invertible layer 의 출력인지 확인
            x.release_saved_output()  # ctx.saved_tensors 에 있는 레퍼런스 삭제
            ctx.save_output = x.save_output
        ctx.requires_output = hasattr(x, "release_saved_output")  # 다음 레이어야 output 을 저장해야하는지 확인

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)
        del x


        Y_1 = X_1 + F(X_2)

        # free memory since X_1 is now not needed
        del X_1

        Y_2 = X_2 + G(Y_1)

        # free memory since X_2 is now not needed
        del X_2
        
        output = torch.cat([Y_1, Y_2], dim=-1)
        del Y_1, Y_2
        # output.save_for_backward = ctx.save_for_backward  ## 이러면 forward 할때 output이 free 될수 있나?
        ## 일단 저장하고, 뒤에 오는게 InvertibleCouplingLayer 타입이면 직접 해제하는 식으로 처리?
        ## 근데 기본적으로 해제될텐데? explicit 하게 저장하도록 하는게 맞는듯
        ## forward 할때 default 는 해제임.
        ## 다음 레이어가 InvertibleCouplingLayer 타입이 아닌지 체크하고, 아니라면 저장하도록 하는게 맞는듯
        ## invertible layer 가 아니면 save_for_backward 호출하도록 하는 방법..?
        # 입력이 invertible 레이어에서 온 것인지 확인
        ctx.output = output.detach()

        def release_saved_output():
            del ctx.output
            # delattr(ctx, 'output')
        def save_output(x):
            ctx.output = x
        output.release_saved_output = release_saved_output
        output.save_output = save_output
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        y = ctx.output
        F, G = ctx.F, ctx.G

        # obtaining gradients dX_1 and dX_2 from the concatenated input
        Y_1, Y_2 = torch.chunk(y, 2, dim=-1)
        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad_(True)

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1 = G(Y_1)
            g_Y_1.backward(dY_2)
        Y_1_grad = Y_1.grad
        Y_1 = Y_1.detach()
        g_Y_1 = g_Y_1.detach()

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():
            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dX_1 = dY_1 + Y_1_grad

            # free memory since Y_1.grad is now not needed
            del Y_1_grad

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad_(True)

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2 = F(X_2)
            f_X_2.backward(dX_1)
        X_2_grad = X_2.grad
        X_2 = X_2.detach()
        f_X_2 = f_X_2.detach()

        with torch.no_grad():
            dX_2 = dY_2 + X_2_grad
            del X_2_grad

        dx = torch.cat([dX_1, dX_2], dim=-1)

        ## 여기서 다음 backward pass 에 X_1, X_2 를 전달해줘야함
        if ctx.requires_output:
            X_1 = Y_1 - f_X_2
            del f_X_2
            x = torch.cat([X_1, X_2], dim=-1)
            ctx.save_output(x.detach())
        return dx, None, None, None


# class MomentumResidualBlock(nn.Module):

#     def __init__(self, F: nn.Module, gamma = 0.1, invert_when_backward=True):
#         super().__init__()
#         self.F = F
#         self.invert_when_backward = invert_when_backward

#     def forward(self, x):
#         if self.invert_when_backward:
#             return InvertibleMomentumResidualLayer.apply(x, self.)
#         else:
#             X, V = torch.chunk(x, 2, dim=-1)
#             V = V + self.F(X)
#             Y_2 = X_2 + self.G(Y_1)
#             return torch.cat([Y_1, Y_2], dim=-1)


def finite_diff_grad_check_couplingblock():
    # device = torch.device("cuda")
    device = torch.device("cpu")
    input = torch.rand(1, 16, requires_grad=True, dtype=torch.float64, device=device)

    num_blocks = 10
    mlp = lambda: nn.Sequential(
        nn.LayerNorm(8),
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8)
    )
    model = nn.Sequential(*[
        CouplingBlock(mlp(), mlp())
        for _ in range(num_blocks)
    ])

    model.to(dtype=torch.float64, device=device)
    
    # @torch.autocast("cuda", dtype=torch.bfloat16)
    # @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def forward_loss_fn(x):
        x = model(x)
        loss = x.norm()
        return loss

    if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
        print("Gradient check passed!")


def finite_diff_grad_check_residualblock():
    # device = torch.device("cuda")
    device = torch.device("cpu")
    input = torch.rand(1, 16, requires_grad=True, dtype=torch.float64, device=device)

    num_blocks = 10
    # mlp = lambda: nn.Sequential(
    #     nn.LayerNorm(8),
    #     nn.Linear(8, 8),
    #     nn.GELU(),
    #     nn.Linear(8, 8)
    # )
    # model = nn.Sequential(*[
    #     CouplingBlock(mlp(), mlp())
    #     for _ in range(num_blocks)
    # ])


    mlp = lambda: nn.Sequential(
        # nn.LayerNorm(8),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
    )
    model = nn.Sequential(*[
        ResidualBlock(mlp())
        for _ in range(num_blocks)
    ])

    model.to(dtype=torch.float64, device=device)
    
    # @torch.autocast("cuda", dtype=torch.bfloat16)
    # @torch.cuda.amp.autocast(dtype=torch.bfloat16)


    def forward_loss_fn(x):
        x = model(x)
        loss = x.norm()
        return loss

    if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
        print("Gradient check passed!")


if __name__ == "__main__":
    finite_diff_grad_check_couplingblock()
