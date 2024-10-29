# invertible-nn
Yet another invertible neural nets

## Installation
```
pip install git+https://github.com/hushon/invertible-nn.git
```

## Examples
The basic building block is the `invertible_nn.layers.CouplingBlock` which implements the reversible layer using coupling function.  
Coupling function consists of an arbitrary function F and G which performs following transform:  
$$X_1, X_2 \leftarrow \text{split}(X)$$  
$$Y_1 = X_1 + F(X_2)$$  
$$Y_2 = X_2 + G(Y_1)$$  
$$Y \leftarrow [Y_1, Y_2]$$  

Because this function is reversible, the intermediate states can be reconstructed during the backward pass instead of being stored in the memory. 

### Reversible ViT
A reversible Vision Transformer architecture is implemented based on the coupling block. 

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