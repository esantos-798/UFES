import torch.nn as nn

from src.models.base_lstnet import BaseLSTNet
from src.models.components import SelfAttention


class AttentionLSTNet(BaseLSTNet):

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

        self.attention = SelfAttention(hidden_size)

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(self,x):

        seq = self.extract_features(x)

        context = self.attention(seq)

        context = self.dropout(context)

        return self.fc(context)