from .preprocessing import OpticalFaultPreprocessor
from .dataset import OpticalFaultDataset
from .dataloader import get_dataloader
from config import HARD_FAILURE_DATASET

import numpy as np


processor = OpticalFaultPreprocessor(
    HARD_FAILURE_DATASET
)


X, y, failure = processor.run()


dataset = OpticalFaultDataset(
    X,
    y,
    failure
)


loader = get_dataloader(
    dataset,
    batch_size=64
)


print("Número de batches:")
print(len(loader))


for batch_X, batch_y, batch_failure in loader:

    print("\nBatch X:")
    print(batch_X.shape)

    print("\nBatch y:")
    print(batch_y.shape)

    print("\nBatch failure:")
    print(batch_failure.shape)

    break

print(
    np.unique(
        failure,
        return_counts=True
    )
)