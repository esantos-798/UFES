import torch

from .lstm import LSTMModel

INPUT_SIZE = 12
HIDDEN_SIZE = 100
NUM_LAYERS = 1
OUTPUT_SIZE = 12


model = LSTMModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE
)


print(model)


x = torch.randn(
    64,
    30,
    12
)


y = model(x)


print("\nEntrada:")
print(x.shape)


print("\nSaída:")
print(y.shape)