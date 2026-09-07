import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):

    def __init__(
        self,
        input_size=12,
        hidden_size=100,
        num_layers=1
    ):

        super().__init__()


        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )


    def forward(self,x):

        _,(hidden,cell)=self.lstm(x)

        features = hidden[-1]

        return features