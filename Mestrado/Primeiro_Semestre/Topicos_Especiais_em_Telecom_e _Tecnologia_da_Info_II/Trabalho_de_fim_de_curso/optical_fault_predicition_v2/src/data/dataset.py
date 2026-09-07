import torch
from torch.utils.data import Dataset


class OpticalDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        failure=None,
        task="classification"
    ):

        self.X = torch.FloatTensor(X)

        self.y = torch.FloatTensor(y)

        self.failure = (
            None
            if failure is None
            else torch.FloatTensor(failure)
        )

        self.task = task

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        if self.task == "classification":

            return (
                self.X[idx],
                self.y[idx],
                self.failure[idx]
            )

        elif self.task == "forecast":

            return (
                self.X[idx],
                self.y[idx]
            )

        else:

            raise ValueError(
                f"Unknown task: {self.task}"
            )