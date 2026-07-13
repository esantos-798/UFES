import torch
import torch.nn as nn


class CNNLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        cnn_channels=32,
        kernel_size=3,
        num_layers=1,
        dropout=0.2
    ):

        super().__init__()


        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_channels,
            kernel_size=kernel_size
        )


        self.relu = nn.ReLU()


        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )


        self.dropout = nn.Dropout(dropout)


        self.fc = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(self, x):

        # x:
        # batch, seq, features
        

        # Conv1d espera:
        # batch, channels, seq

        x = x.permute(
            0,
            2,
            1
        )


        x = self.conv(x)

        x = self.relu(x)


        # volta:
        # batch, seq, channels

        x = x.permute(
            0,
            2,
            1
        )


        output, (hidden, cell) = self.lstm(x)


        x = hidden[-1]


        x = self.dropout(x)


        out = self.fc(x)


        return out