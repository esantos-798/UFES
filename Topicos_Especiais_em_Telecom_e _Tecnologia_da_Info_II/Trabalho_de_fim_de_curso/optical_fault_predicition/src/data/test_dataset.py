from .preprocessing import OpticalFaultPreprocessor
from .dataset import OpticalFaultDataset
from config import HARD_FAILURE_DATASET


processor = OpticalFaultPreprocessor(
    HARD_FAILURE_DATASET
)


X, y, failure = processor.run()


dataset = OpticalFaultDataset(
    X,
    y,
    failure
)

print(dataset)

print("Número de amostras:")
print(len(dataset))


sample_X, sample_y = dataset[0]


print("\nEntrada:")
print(sample_X.shape)

print("\nSaída:")
print(sample_y.shape)