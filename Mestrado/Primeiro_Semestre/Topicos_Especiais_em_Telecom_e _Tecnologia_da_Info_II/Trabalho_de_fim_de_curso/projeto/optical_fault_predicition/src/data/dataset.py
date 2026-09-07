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


        # ============================
        # Dados de entrada
        # ============================

        self.X = torch.FloatTensor(
            X
        )


        self.task = task



        # ============================
        # Label de falha
        # ============================

        if failure is not None:


            self.failure = torch.FloatTensor(
                failure
            )


        else:

            self.failure = torch.zeros(
                len(X)
            )



        # ============================
        # Target
        # ============================

        if task == "classification":


            # classificação:
            # prever se haverá falha

            self.y = self.failure.unsqueeze(1)



        else:


            # forecast:
            # prever valores futuros

            self.y = torch.FloatTensor(
                y
            )



    # ============================
    # tamanho dataset
    # ============================

    def __len__(self):

        return len(
            self.X
        )



    # ============================
    # item
    # ============================

    def __getitem__(
        self,
        idx
    ):


        return (

            self.X[idx],

            self.y[idx],

            self.failure[idx]

        )