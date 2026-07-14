import torch
import torch.nn as nn


class AttentionLSTNet(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=100,
        output_size=12,
        cnn_channels=32,
        kernel_size=3,
        dropout=0.2
    ):

        super().__init__()


        self.hidden_size = hidden_size


        # CNN
        self.conv = nn.Conv2d(
            1,
            cnn_channels,
            kernel_size=(kernel_size,input_size)
        )


        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )


        # GRU temporal

        self.gru = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )


        # Attention

        self.attention = nn.Linear(
            hidden_size,
            1
        )


        # AR branch

        self.ar = nn.Linear(
            30 * input_size,
            output_size
        )


        # Final

        self.fc = nn.Linear(
            hidden_size + output_size,
            output_size
        )



    def forward(self,x):


        batch = x.size(0)


        # CNN

        c = x.unsqueeze(1)


        c = self.conv(c)

        c = self.relu(c)

        c = self.dropout(c)


        # remove last dim

        c = c.squeeze(3)


        # (batch, channels,time)

        c = c.permute(
            0,
            2,
            1
        )


        # GRU

        out,_ = self.gru(c)


        # Attention

        weights = self.attention(out)


        weights = torch.softmax(
            weights,
            dim=1
        )


        context = torch.sum(
            weights * out,
            dim=1
        )


        # AR

        ar_input = x.reshape(
            batch,
            -1
        )


        ar = self.ar(
            ar_input
        )


        # Fusion

        result = torch.cat(
            [
                context,
                ar
            ],
            dim=1
        )


        return self.fc(result)