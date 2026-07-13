import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(
        self,
        input_size=12,
        hidden_size=100,
        num_layers=1,
        output_size=12
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers


        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )


        self.fc = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        # último estado temporal
        x = hidden[-1]

        x = self.fc(x)

        return x