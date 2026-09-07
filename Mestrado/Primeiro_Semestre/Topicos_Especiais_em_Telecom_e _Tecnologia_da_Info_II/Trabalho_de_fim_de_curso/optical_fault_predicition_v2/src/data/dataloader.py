from torch.utils.data import DataLoader
from torch.utils.data import random_split

from .dataset import OpticalDataset
from .preprocessing import OpticalFaultPreprocessor

from config import HARD_FAILURE_DATASET


def get_dataloader(
    batch_size=64,
    shuffle=True,
    task="classification",
    dataset_path=HARD_FAILURE_DATASET
):

    processor = OpticalFaultPreprocessor(
        dataset_path
    )

    X, y, failure = processor.run()

    dataset = OpticalDataset(
        X,
        y,
        failure=failure,
        task=task
    )

    print("X:", X.shape)
    print("y:", y.shape)

    if task == "classification":
        print("failure:", failure.shape)

    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader