import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

from typing import Callable, Tuple, Any


class InvertibleLayer(Function):
    @staticmethod
    def forward(ctx, x):
        if hasattr(x, "save_for_backward"):
            ctx.save_for_backward = x.save_for_backward
            delattr(x, "save_for_backward")


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
        """
        ctx.F = F
        ctx.G = G
        if hasattr(x, "release_saved_output"):  ## invertible layer 의 출력인지 확인
            x.release_saved_output()  # ctx.saved_tensors 에 있는 레퍼런스 삭제
        ctx.requires_output = hasattr(x, "release_saved_output")  # 다음 레이어야 output 을 저장해야하는지 확인

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        """
        forward pass equations:
        Y_1 = X_1 + F(X_2), F = Attention
        Y_2 = X_2 + G(Y_1), G = MLP
        """

        Y_1 = X_1 + F(X_2)

        # free memory since X_1 is now not needed
        del X_1

        Y_2 = X_2 + G(Y_1)

        # free memory since X_2 is now not needed
        del X_2
        
        output = torch.cat([Y_1, Y_2], dim=-1)
        # output.save_for_backward = ctx.save_for_backward  ## 이러면 forward 할때 output이 free 될수 있나?
        ## 일단 저장하고, 뒤에 오는게 InvertibleCouplingLayer 타입이면 직접 해제하는 식으로 처리?
        ## 근데 기본적으로 해제될텐데? explicit 하게 저장하도록 하는게 맞는듯
        ## forward 할때 default 는 해제임.
        ## 다음 레이어가 InvertibleCouplingLayer 타입이 아닌지 체크하고, 아니라면 저장하도록 하는게 맞는듯
        ## invertible layer 가 아니면 save_for_backward 호출하도록 하는 방법..?
        # 입력이 invertible 레이어에서 온 것인지 확인
        ctx.output = output

        def release_saved_output():
            ctx.output = None
        output.release_saved_tensors = release_saved_output
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
        if ctx.output is not None:
            y = ctx.output
        else:
            y = dy.output
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
            dX_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad_(True)

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2 = F(X_2)
            f_X_2.backward(dX_1)
        
        with torch.no_grad():
            dX_2 = dY_2 + X_2.grad

        dx = torch.cat([dX_1, dX_2], dim=-1)

        ## 여기서 다음 backward pass 에 X_1, X_2 를 전달해줘야함
        if ctx.requires_output:
            X_1 = Y_1 - f_X_2
            x = torch.cat([X_1, X_2], dim=-1)
            dx.output = x.detach().clone()

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
            ## prev_ctx 에 output 넣어주는걸 여기로 옮겨도 괜찮을듯
            ## ctx 가 autograd 인터페이스 밖으로 나오는게 좋지않음.. 메모리 free 안될수도 있음.
            ## 잘하면 output 에 backward hook 을 이용해서 ctx 를 숨길수도 있을듯 한데..
            ## 아님 외부에 context manager 를 둬서 거기다가 ctx 를 숨기는 방법도 있을듯
            ## with invertible_forward(enabled=True): 이런식으로
            ## 근데 그럼 다음 레이어가 invertible 한지를 알아야함 (input 을 되돌려줄수있는지)
        else:
            X_1, X_2 = torch.chunk(x, 2, dim=-1)
            Y_1 = X_1 + self.F(X_2)
            Y_2 = X_2 + self.G(Y_1)
            return torch.cat([Y_1, Y_2], dim=-1)


def finite_diff_grad_check():
    device = torch.device("cuda")
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

    if torch.autograd.gradcheck(forward_loss_fn, input):
        print("Gradient check passed!")


if __name__ == "__main__":
    finite_diff_grad_check()
