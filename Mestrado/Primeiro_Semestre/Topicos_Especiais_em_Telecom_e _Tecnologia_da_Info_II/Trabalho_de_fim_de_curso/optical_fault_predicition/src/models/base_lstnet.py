import torch
import torch.nn as nn


class BaseLSTNet(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        cnn_channels=32,
        kernel_size=3
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            1,
            cnn_channels,
            (kernel_size, input_size)
        )

        self.gru = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )

    def extract_features(self, x):

        x = x.unsqueeze(1)

        x = torch.relu(self.conv(x))

        x = x.squeeze(3)

        x = x.permute(0, 2, 1)

        seq, _ = self.gru(x)

        return seq