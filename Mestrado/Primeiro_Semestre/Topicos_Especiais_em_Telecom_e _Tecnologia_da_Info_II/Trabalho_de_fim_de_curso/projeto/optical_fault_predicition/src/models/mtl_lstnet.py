import torch
import torch.nn as nn

class MTLLSTNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        cnn_channels=32,
        kernel_size=3,
        skip=5,
        dropout=0.2
    ):
        super().__init__()
        self.output_size = output_size
        self.skip = skip
        self.hidden_size = hidden_size

        ####################################################
        # CNN temporal
        ####################################################
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels,
            kernel_size=(kernel_size, input_size)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        ####################################################
        # Recurrent encoders
        ####################################################
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.skip_gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        ####################################################
        # AR Component
        ####################################################
        self.ar = nn.Linear(input_size * 30, output_size)

        ####################################################
        # Forecast branch
        ####################################################
        self.forecast_head = nn.Linear(
            hidden_size * 2,
            output_size
        )

        ####################################################
        # Failure classification branch
        ####################################################
        self.failure_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ####################################################
        # x: batch, window, features
        ####################################################
        batch = x.size(0)
        c = x.unsqueeze(1)
        c = self.conv(c)
        c = self.relu(c)
        c = self.dropout(c)
        c = c.squeeze(3)
        c = c.permute(0, 2, 1).contiguous()
        seq_len_c = c.size(1)

        ####################################################
        # Main RNN
        ####################################################
        rnn_out, _ = self.gru(c)
        rnn_last = rnn_out[:, -1, :]

        ####################################################
        # Skip RNN
        ####################################################
        if seq_len_c >= self.skip:
            length = seq_len_c // self.skip
            s = c[:, -length * self.skip:, :]
            s = s.view(batch, length, self.skip, c.size(2))
            s = s.transpose(1, 2).contiguous()
            s = s.view(batch * self.skip, length, c.size(2))
            
            skip_out, _ = self.skip_gru(s)
            skip_last = skip_out[:, -1, :]
            skip_last = skip_last.view(batch, self.skip, self.hidden_size)
            skip_last = skip_last.mean(dim=1)
        else:
            skip_last = torch.zeros(batch, self.hidden_size, device=x.device)

        context = torch.cat([rnn_last, skip_last], dim=1)

        ####################################################
        # Multi-task heads
        ####################################################
        ar_input = x.view(batch, -1)
        ar_out = self.ar(ar_input)
        
        forecast = self.forecast_head(context) + ar_out
        failure = self.failure_head(context)

        return {
            "forecast": forecast,
            "failure": failure
        }