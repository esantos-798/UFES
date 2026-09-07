import torch.nn as nn

from src.models.base_lstnet import BaseLSTNet


class LSTNet(BaseLSTNet):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        cnn_channels=32,
        kernel_size=3,
        dropout=0.2
    ):

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            cnn_channels=cnn_channels,
            kernel_size=kernel_size
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        seq = self.extract_features(x)

        last_hidden = seq[:, -1, :]

        last_hidden = self.dropout(last_hidden)

        out = self.fc(last_hidden)

        return out