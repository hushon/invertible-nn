# invertible-nn
Yet another invertible neural nets.  
This implements building blocks for reversible neural net. Because the layers are reversible, we can avoid caching the intermediate activations to the memory and instead reconstruct them during the backward pass. 

## Installation
```
pip install git+https://github.com/hushon/invertible-nn.git
```

## Examples

### CouplingBlock
`invertible_nn.layers.CouplingBlock` implements the reversible layer using coupling function.  
Coupling function consists of an arbitrary function F and G which performs following transform:  
$$X_1, X_2 \leftarrow \text{split}(X)$$  
$$Y_1 = X_1 + F(X_2)$$  
$$Y_2 = X_2 + G(Y_1)$$  
$$Y \leftarrow [Y_1, Y_2]$$  

Typically, F and G can be a small neural network such as an MLP or a self-attention layer.

### Reversible ViT
A reversible Vision Transformer architecture is implemented by composing the `CouplingBlock` layers. 

```python
from invertible_nn.invertible_vit import InvertibleVisionTransformer

device = torch.device("cuda")

model = InvertibleVisionTransformer(
    depth=12,
    patch_size=(16, 16),
    image_size=(224, 224),
    num_classes=1000
)
model.to(device=device)

input = torch.rand(128, 3, 224, 224, device=device)
output = model(input)
loss = output.norm()
loss.backward()
```

### ResidualBlock

`invertible_nn.layers.ResidualBlock` implements a reversible residual layer.  
This block consists of an arbitrary function F and performs the following transform:  
$$y = x + F(x)$$

F(x) is required to be a 1-Lipschitz function for reversibility. 
We use an MLP and apply spectral normalization to enforce the Lipschitz constraint. 
During backward pass, the input is reconstructed using fixed-point iteration method. 