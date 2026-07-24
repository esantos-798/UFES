import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    """
    Remove padding extra à direita para manter a convolução estritamente causal.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x) if self.downsample else x
        return self.final_relu(out + res)


class TCN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,  # Tornou-se opcional caso usada apenas como extrator
        num_levels=3,
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        layers = []
        channels = input_size

        # Permite passar hidden_size como lista [64, 64, 64] ou como um único int
        if isinstance(hidden_size, int):
            level_channels = [hidden_size] * num_levels
        else:
            level_channels = hidden_size
            num_levels = len(level_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    channels,
                    level_channels[i],
                    kernel_size,
                    dilation,
                    dropout
                )
            )
            channels = level_channels[i]

        self.network = nn.Sequential(*layers)
        
        # Cabeça linear opcional (caso queira usá-la direto sem um decodificador)
        self.fc = nn.Linear(channels, output_size) if output_size is not None else None

    def forward(self, x, return_sequence=True):
        # x shape esperado vindo do dataloader: [Batch, Seq_Len, Input_Size]
        
        # Permuta para o formato esperado por convoluções 1D: [Batch, Features, Seq_Len][cite: 9]
        x = x.permute(0, 2, 1)
        
        # Passa pela rede convolucional causal
        y = self.network(x)  # Out shape: [Batch, Hidden_Size, Seq_Len]
        
        # Retorna para o formato padrão do PyTorch para sequências: [Batch, Seq_Len, Hidden_Size]
        y = y.permute(0, 2, 1)

        if return_sequence:
            # ESSENCIAL PARA FORECASTING: Preserva a dimensão temporal completa
            if self.fc is not None:
                return self.fc(y)
            return y
            
        # CLASSIFICAÇÃO / DETECÇÃO: Extrai apenas o último passo temporal (o "resumo" do passado)[cite: 9]
        last_step = y[:, -1, :] 
        if self.fc is not None:
            return self.fc(last_step)
        return last_step