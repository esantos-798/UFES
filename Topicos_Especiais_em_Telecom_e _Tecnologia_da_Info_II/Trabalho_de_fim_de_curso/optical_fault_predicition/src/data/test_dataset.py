from src.data.dataset import OpticalDataset
from src.data.preprocessing import OpticalFaultPreprocessor
from config import HARD_FAILURE_DATASET


processor = OpticalFaultPreprocessor(
    HARD_FAILURE_DATASET
)


X, y, failure = processor.run()


dataset = OpticalDataset(
    X,
    y,
    failure,
    task="classification"
)


sample = dataset[0]


print("X:", sample[0].shape)
print("target:", sample[1].shape)
print("target:", sample[1])