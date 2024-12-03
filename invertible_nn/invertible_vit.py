import torch
from torch import nn
import invertible_nn
import math


class Adapter(nn.Module):
    def __init__(self,
                 n_embed,
                 down_size,
                 dropout=0.0,
                 adapter_scalar="learnable_scalar",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = n_embed
        self.down_size = down_size

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            # nn.init.normal_(self.down_proj.weight, std=0.02)
            # nn.init.normal_(self.up_proj.weight, std=0.02)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        output = up

        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, r=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.adapter = Adapter(n_embed=dim, down_size=dim // r, dropout=0.0, adapter_scalar="1.0", adapter_layernorm_option="in")

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        residual = residual + x
        x = self.norm2(residual)
        out1 = self.mlp(x)
        out2 = self.adapter(x)
        return residual + out1 + out2


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, r=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.adapter = Adapter(n_embed=dim, down_size=dim // r, dropout=0.0, adapter_scalar="1.0", adapter_layernorm_option="in")

    def forward(self, x):
        residual = x
        x = self.norm(x)
        out1, _ = self.attn(x, x, x)
        out2 = self.adapter(x)
        return residual + out1 + out2


class MLPBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, r=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.adapter = Adapter(n_embed=dim, down_size=dim//r, dropout=0.0, adapter_scalar="1.0", adapter_layernorm_option="in")

    def forward(self, x):
        residual = x
        x = self.norm(x)
        out1 = self.mlp(x)
        out2 = self.adapter(x)
        return residual + out1 + out2


class MEFT_Block1(invertible_nn.scaled_coupling_layer.ScaledCouplingBlock):
    def __init__(self, embed_dim, n_head, reduction_ratio):
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.reduction_ratio = reduction_ratio
        self.x1_factor = 0.1
        self.x2_factor = 1.0

        transformer_block = TransformerBlock(dim=self.embed_dim, num_heads=self.n_head, mlp_ratio=4, r=self.reduction_ratio)
        adapter = MLPBlock(dim=self.embed_dim, mlp_ratio=4, r=self.reduction_ratio)

        super().__init__(
            F=transformer_block,
            G=adapter,
            X1_factor=self.x1_factor,
            X2_factor=self.x2_factor,
            switch=True
        )


class MEFT_Block2(invertible_nn.scaled_coupling_layer.ScaledCouplingBlock):
    def __init__(self, embed_dim, n_head, reduction_ratio):
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.reduction_ratio = reduction_ratio
        self.x1_factor = 1.0
        self.x2_factor = 0.1

        transformer_block = TransformerBlock(dim=self.embed_dim, num_heads=self.n_head, mlp_ratio=4, r=self.reduction_ratio)
        adapter = MLPBlock(dim=self.embed_dim, mlp_ratio=4, r=self.reduction_ratio)

        super().__init__(
            F=transformer_block,
            G=adapter,
            X1_factor=self.x1_factor,
            X2_factor=self.x2_factor,
            switch=True
        )


class MEFT_Block3(invertible_nn.scaled_coupling_layer.ScaledCouplingBlock):
    def __init__(self, embed_dim, n_head, reduction_ratio):
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.reduction_ratio = reduction_ratio
        self.x1_factor = 0.1
        self.x2_factor = 0.1

        attention = AttentionBlock(dim=self.embed_dim, num_heads=self.n_head, r=self.reduction_ratio)
        mlp = MLPBlock(dim=self.embed_dim, mlp_ratio=4, r=self.reduction_ratio)

        super().__init__(
            F=attention,
            G=mlp,
            X1_factor=self.x1_factor,
            X2_factor=self.x2_factor,
            switch=True
        )


class InvertibleVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        n_head=8,
        depth=8,
        patch_size=(2, 2),  # this patch size is used for CIFAR-10 --> (32 // 2)**2 = 256 sequence length
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size

        num_patches = (image_size[0] // self.patch_size[0]) * \
            (image_size[1] // self.patch_size[1])

        self.layers = nn.ModuleList(
            [
                MEFT_Block1(
                    dim=self.embed_dim,
                    num_heads=self.n_head
                )
                for _ in range(self.depth)
            ]
        )

        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim)
        )

        self.head = nn.Linear(2 * self.embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(2 * self.embed_dim)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embeddings

        x1, x2 = x, x
        output_hooks = None
        for layer in self.layers:
            x1, x2, output_hooks = layer(x1, x2, output_hooks)
        
        x = torch.cat([x1, x2], dim=-1)

        # aggregate across sequence length
        x = x.mean(1)

        # head pre-norm
        x = self.norm(x)

        # pre-softmax logits
        x = self.head(x)

        # return pre-softmax logits
        return x


def test():
    device = torch.device("cuda")
    input = torch.rand(1, 3, 224, 224, requires_grad=True, device=device)

    model = InvertibleVisionTransformer(
        depth=30,
        patch_size=(16, 16),
        image_size=(224, 224),
        num_classes=1000
    )
    model.to(device=device)

    import numpy as np
    print(sum([np.prod(p.size()) for p in model.parameters()]))

    # # @torch.autocast("cuda", dtype=torch.bfloat16)
    # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
    #     output = model(input)
    #     loss = output.norm()
    # loss.backward()
    # breakpoint()


    # @torch.autocast("cuda", dtype=torch.bfloat16)
    # @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def forward_loss_fn(x):
        x = model(x)
        loss = x.norm()
        return loss
    
    breakpoint()
    for module in model.modules():
        if isinstance(module, InvertibleTransformerBlock):
            module.invert_when_backward = False
    breakpoint()


    if torch.autograd.gradcheck(forward_loss_fn, input, nondet_tol=1e-5):
        print("Gradient check passed!")


if __name__ == "__main__":
    test()

