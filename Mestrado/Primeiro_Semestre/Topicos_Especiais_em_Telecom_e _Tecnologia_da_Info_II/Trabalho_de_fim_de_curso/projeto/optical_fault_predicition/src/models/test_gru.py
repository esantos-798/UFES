import torch

from .gru import GRUModel

model = GRUModel(
    input_size=12,
    hidden_size=100,
    num_layers=1,
    output_size=1
)

print(model)

x = torch.randn(64, 30, 12)

y = model(x)

print(y.shape)