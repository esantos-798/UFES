import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLSTNet(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.2
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            1,
            32,
            kernel_size=(3, input_size)
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=32,
            hidden_size=hidden_size,
            batch_first=True
        )

        # -------- Forecast Head --------

        self.forecast_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_size,
                output_size
            )

        )

        # -------- Failure Head --------

        self.failure_head = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size // 2
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_size // 2,
                1
            ),

            nn.Sigmoid()

        )


    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = x.squeeze(3)

        x = x.permute(0, 2, 1)

        output, _ = self.gru(x)

        context = output[:, -1]

        forecast = self.forecast_head(context)

        failure = self.failure_head(context)

        return forecast, failure