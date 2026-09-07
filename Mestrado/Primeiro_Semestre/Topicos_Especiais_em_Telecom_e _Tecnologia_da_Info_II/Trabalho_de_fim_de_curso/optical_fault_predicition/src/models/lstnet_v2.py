import torch
import torch.nn as nn

class LSTNetV2(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=100,
        cnn_channels=32,
        kernel_size=3,
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
        # RNN principal
        # =========================
        self.gru = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )

        # =========================
        # Skip RNN
        # =========================
        self.skip_gru = nn.GRU(
            cnn_channels,
            hidden_size,
            batch_first=True
        )

        # =========================
        # AR component
        # =========================
        # Assuming sequence length 30 as defined in datasets
        self.ar = nn.Linear(
            input_size * 30,
            output_size
        )

        # =========================
        # Fusion
        # =========================
        self.fc = nn.Linear(
            hidden_size * 2,
            output_size
        )

    def forward(self, x):
        batch = x.size(0)

        # x: batch, seq, features
        # CNN espera: batch, channel, seq, features
        c = x.unsqueeze(1)
        c = self.conv(c)
        c = self.relu(c)
        c = self.dropout(c)

        # batch, channels, seq
        c = c.squeeze(3)

        # batch, seq, channels
        c = c.transpose(1, 2).contiguous()
        seq_len_c = c.size(1)

        # =====================
        # Main GRU
        # =====================
        rnn_out, _ = self.gru(c)
        rnn_last = rnn_out[:, -1, :]

        # =====================
        # Skip GRU real (Corrigido)
        # =====================
        if seq_len_c >= self.skip:
            length = seq_len_c // self.skip
            
            # Pega as ultimas amostras de tamanho multiplo de "skip"
            s = c[:, -length * self.skip:, :]
            
            # Realiza o reshape para processar os "skip" passos temporalmente de forma intercalada
            s = s.view(batch, length, self.skip, c.size(2))
            s = s.transpose(1, 2).contiguous()  # [batch, skip, length, channels]
            s = s.view(batch * self.skip, length, c.size(2))  # [batch * skip, length, channels]
            
            skip_out, _ = self.skip_gru(s)
            skip_last = skip_out[:, -1, :]  # [batch * skip, hidden_size]
            
            # Reverte o formato e combina a resposta dos ramos de skip
            skip_last = skip_last.view(batch, self.skip, self.hidden_size)
            skip_last = skip_last.mean(dim=1)  # [batch, hidden_size]
        else:
            skip_last = torch.zeros(batch, self.hidden_size, device=x.device)

        # =====================
        # AR
        # =====================
        ar = x.view(batch, -1)
        ar = self.ar(ar)

        # =====================
        # Fusion
        # =====================
        neural = torch.cat(
            [
                rnn_last,
                skip_last
            ],
            dim=1
        )
        neural = self.fc(neural)

        output = neural + ar

        return output