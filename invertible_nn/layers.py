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
        """

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
        
        ctx.F = F
        ctx.G = G
        if hasattr(x, "ctx"):
            ctx.prev_ctx = x.ctx # 이전 레이어한테 input 복원한거 보내줘야하므로 필요

        output = torch.cat([Y_1, Y_2], dim=-1)
        output.ctx = ctx  ## 이러면 forward 할때 output이 free 될수 있나?

        ## output 다음에 오는 함수가 invertible 한지 알 수 있나?
        ## invertible 하다면 output 을 free 하고 아니면 ctx.save_for_backward 호출하도록

        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dy: torch.Tensor) -> torch.Tensor:
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        y = ctx.to_save[0]  ## next block backward 때 또는 마지막 레이어 끝나고 사람이 넣어줌
        F, G = ctx.F, ctx.G

        # obtaining gradients dX_1 and dX_2 from the concatenated input
        Y_1, Y_2 = torch.chunk(y, 2, dim=-1)
        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

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
            X_2.requires_grad = True

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
        if hasattr(ctx, "prev_ctx"):
            X_1 = Y_1 - f_X_2
            x = torch.cat([X_1, X_2], dim=-1)
            ctx.prev_ctx.save_for_backward(x.detach().clone())

        return dx, None, None, None


class CouplingBlock(nn.Module):
    """
    F 랑 G 는 임의의 모듈
    F랑 G를 coupling 구조에 끼워넣음. 
    backward pass 할때는 뒷쪽 블락에서 보내준 activation 값을 이용해 중간값 재계산
    Y_1 = X_1 + F(X_2)
    Y_2 = X_2 + G(Y_1)
    """
    def __init__(self, F: nn.Module, G: nn.Module, use_invertible=True):
        super().__init__()
        self.F = F
        self.G = G
        self.use_invertible = use_invertible

    def forward(self, x):
        if self.use_invertible:
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
    input = torch.rand(1, 16, requires_grad=True, dtype=torch.float64)
    model = nn.Sequential(
        CouplingBlock(
            nn.Linear(8, 8),
            nn.Linear(8, 8),
        ),
        CouplingBlock(
            nn.Linear(8, 8),
            nn.Linear(8, 8),
        ),
    )
    model.to(torch.float64)
    
    def forward_loss_fn(x):
        x = model(x)
        if hasattr(x, "ctx"):
            x.ctx.save_for_backward(x.detach().clone())
        loss = x.norm()
        return loss
    
    if torch.autograd.gradcheck(forward_loss_fn, input):
        print("Gradient check passed!")


if __name__ == "__main__":
    finite_diff_grad_check()
