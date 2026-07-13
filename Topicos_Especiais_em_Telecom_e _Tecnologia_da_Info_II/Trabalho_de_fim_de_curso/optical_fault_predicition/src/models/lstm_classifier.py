import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )


        self.fc = nn.Linear(
            hidden_size,
            1
        )


    def forward(self,x):

        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        prediction = self.fc(
            last_hidden
        )


        return prediction