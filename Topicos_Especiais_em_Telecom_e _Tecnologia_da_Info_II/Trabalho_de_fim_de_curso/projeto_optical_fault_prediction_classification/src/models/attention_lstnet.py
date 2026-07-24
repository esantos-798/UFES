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
        skip=5,
        dropout=0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.skip = skip
        self.output_size = output_size

        # CNN
        self.conv = nn.Conv2d(
            1,
            cnn_channels,
            kernel_size=(kernel_size, input_size)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # GRU temporal
        self.gru = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )
        
        # Skip GRU temporal
        self.skip_gru = nn.GRU(
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

        # Final (hidden_size from attention context + hidden_size from skip branch)
        self.fc = nn.Linear(
            hidden_size * 2,
            output_size
        )

    def forward(self, x):
        batch = x.size(0)

        # CNN
        c = x.unsqueeze(1)
        c = self.conv(c)
        c = self.relu(c)
        c = self.dropout(c)
        c = c.squeeze(3)
        c = c.permute(0, 2, 1).contiguous()
        seq_len_c = c.size(1)

        # GRU
        out, _ = self.gru(c)

        # Attention
        weights = self.attention(out)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(weights * out, dim=1) # [batch, hidden_size]
        
        # Skip GRU
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
            
        neural = torch.cat([context, skip_last], dim=1)

        # AR
        ar_input = x.view(batch, -1)
        ar = self.ar(ar_input)

        # Fusion
        result = self.fc(neural) + ar

        return result