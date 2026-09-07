import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        #self.fc = nn.Linear(
        #    hidden_size,
        #    output_size
        #)
        self.forecast_head = nn.Linear(
            hidden_size,
            output_size
        )

        self.failure_head = nn.Linear(
            hidden_size,
            1
        )
    def forward(self, x):

        output, hidden = self.gru(x)

        last_hidden = output[:, -1, :]

        #out = self.fc(last_hidden)

        #return out
    
        forecast = self.forecast_head(last_hidden)

        failure = self.failure_head(last_hidden)

        return {
            "forecast": forecast,
            "failure": failure
        }