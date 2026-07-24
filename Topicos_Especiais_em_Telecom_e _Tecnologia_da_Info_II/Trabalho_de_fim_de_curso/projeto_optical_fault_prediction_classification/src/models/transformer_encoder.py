import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Correção segura para garantir compatibilidade caso d_model seja ímpar
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        # Ajuste para evitar mismatch de dimensões se d_model for ímpar
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.position = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_sequence=True):
        # x shape esperado: [Batch, Seq_Len, Input_Size]
        x = self.embedding(x)
        x = self.position(x)
        x = self.encoder(x)
        x = self.dropout(x)
        
        if return_sequence:
            # Mantém [Batch, Seq_Len, d_model] -> Crucial para o Forecasting
            return x
            
        # Retorna [Batch, d_model] -> Opcional para classificação pura
        return torch.mean(x, dim=1)