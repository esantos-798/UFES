import torch
import torch.nn as nn


class LSTNet(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        cnn_channels=32,
        kernel_size=3,
        hidden_size=100,
        dropout=0.2
    ):

        super().__init__()


        self.output_size = output_size


        # =========================
        # CNN temporal
        # =========================

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels,
            kernel_size=(kernel_size, input_size)
        )


        self.relu = nn.ReLU()


        self.dropout = nn.Dropout(
            dropout
        )


        # =========================
        # GRU temporal
        # =========================

        self.gru = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )


        # =========================
        # Skip connection
        # =========================

        self.skip = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )


        # =========================
        # Output
        # =========================

        self.fc = nn.Linear(
            hidden_size * 2,
            output_size
        )



    def forward(self, x):

        batch = x.size(0)


        # x:
        # batch, seq, features

        c = x.unsqueeze(1)


        c = self.conv(c)

        c = self.relu(c)


        c = c.squeeze(3)


        # batch, channels, seq

        c = c.transpose(
            1,
            2
        )


        c = self.dropout(c)



        # =====================
        # RNN
        # =====================

        _, h = self.gru(c)


        h = h[-1]



        # =====================
        # Skip branch
        # =====================

        _, hs = self.skip(c)


        hs = hs[-1]



        # =====================
        # Fusion
        # =====================

        out = torch.cat(
            [
                h,
                hs
            ],
            dim=1
        )


        out = self.fc(out)


        return out