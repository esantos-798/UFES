import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """
    Remove padding extra para manter convolução causal
    """

    def __init__(self, chomp_size):

        super().__init__()

        self.chomp_size = chomp_size


    def forward(self, x):

        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout
    ):

        super().__init__()


        padding = (kernel_size - 1) * dilation


        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )


        self.chomp1 = Chomp1d(
            padding
        )


        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout)


        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )


        self.chomp2 = Chomp1d(
            padding
        )


        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(dropout)


        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,

            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2
        )


        self.downsample = (

            nn.Conv1d(
                in_channels,
                out_channels,
                1
            )

            if in_channels != out_channels

            else None

        )


        self.final_relu = nn.ReLU()



    def forward(self,x):

        out = self.net(x)


        res = x


        if self.downsample:
            res = self.downsample(x)


        return self.final_relu(
            out + res
        )



class TCN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_levels=3,
        kernel_size=3,
        dropout=0.2
    ):

        super().__init__()


        layers = []


        channels = input_size


        for i in range(num_levels):

            dilation = 2 ** i


            layers.append(

                TemporalBlock(

                    channels,

                    hidden_size,

                    kernel_size,

                    dilation,

                    dropout

                )

            )


            channels = hidden_size



        self.network = nn.Sequential(
            *layers
        )


        self.fc = nn.Linear(
            hidden_size,
            output_size
        )


    def forward(self,x):

        # entrada:
        # batch, seq, features


        x = x.permute(
            0,
            2,
            1
        )


        y = self.network(x)


        # último instante temporal

        y = y[:, :, -1]


        out = self.fc(y)


        return out