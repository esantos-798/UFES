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
        skip=5,
        dropout=0.2
    ):
        super().__init__()
        self.output_size = output_size
        self.skip = skip
        self.hidden_size = hidden_size

        # =========================
        # CNN temporal
        # =========================
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=cnn_channels,
            kernel_size=(kernel_size, input_size)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # =========================
        # Main RNN (GRU)
        # =========================
        self.gru = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )

        # =========================
        # Skip RNN (GRU)
        # =========================
        self.skip_gru = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )

        # =========================
        # Autoregressive (AR) Highway
        # =========================
        # In a sequence of length 30, with input_size features
        # Note: In practice, to make this generic, we could use a specific window_size param.
        # Here we assume a sequence length (window_size) of 30 as used in the project.
        self.ar = nn.Linear(input_size * 30, output_size)

        # =========================
        # Fusion Output
        # =========================
        self.fc = nn.Linear(
            hidden_size * 2,
            output_size
        )

    def forward(self, x):
        batch = x.size(0)

        # x shape: [batch, seq_len, features]
        # CNN expects: [batch, in_channels, seq_len, features]
        c = x.unsqueeze(1)
        c = self.conv(c)
        c = self.relu(c)
        c = self.dropout(c)
        
        # Squeeze the feature dimension and transpose for RNN
        # c shape: [batch, channels, seq_len_c]
        c = c.squeeze(3)
        # transpose to: [batch, seq_len_c, channels]
        c = c.transpose(1, 2).contiguous()

        seq_len_c = c.size(1)

        # =====================
        # Main RNN branch
        # =====================
        rnn_out, _ = self.gru(c)
        rnn_last = rnn_out[:, -1, :]  # [batch, hidden_size]

        # =====================
        # Skip RNN branch
        # =====================
        if seq_len_c >= self.skip:
            # We take the maximum length that is a multiple of skip
            length = seq_len_c // self.skip
            
            # Extract only the required tail of the sequence
            s = c[:, -length * self.skip:, :]
            
            # Reshape to create independent sequences for the skip RNN
            # [batch, length, skip, channels]
            s = s.view(batch, length, self.skip, c.size(2))
            
            # Transpose to [batch, skip, length, channels]
            s = s.transpose(1, 2).contiguous()
            
            # Flatten batch and skip dimensions to process in parallel
            # [batch * skip, length, channels]
            s = s.view(batch * self.skip, length, c.size(2))
            
            # Run through skip GRU
            skip_out, _ = self.skip_gru(s)
            
            # Get the last state of each independent sequence
            skip_last = skip_out[:, -1, :]  # [batch * skip, hidden_size]
            
            # Reshape back to [batch, skip, hidden_size] and average across the skip dimension
            skip_last = skip_last.view(batch, self.skip, self.hidden_size)
            skip_last = skip_last.mean(dim=1)  # [batch, hidden_size]
        else:
            skip_last = torch.zeros(batch, self.hidden_size, device=x.device)

        # =====================
        # Fusion
        # =====================
        neural = torch.cat([rnn_last, skip_last], dim=1)
        neural_out = self.fc(neural)

        # =====================
        # Autoregressive (AR) Highway
        # =====================
        ar_input = x.view(batch, -1)  # Flatten temporal and feature dimensions
        ar_out = self.ar(ar_input)

        # Final prediction is the sum of neural and AR outputs
        out = neural_out + ar_out

        return out