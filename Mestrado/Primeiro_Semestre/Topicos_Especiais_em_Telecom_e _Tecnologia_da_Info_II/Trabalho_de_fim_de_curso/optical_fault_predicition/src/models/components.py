import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # x -> (batch, seq, hidden)

        weights = self.attn(x)

        weights = F.softmax(weights, dim=1)

        context = (weights * x).sum(dim=1)

        return context
    
class ResidualConnection(nn.Module):

    def forward(self, x, residual):

        return x + residual

class Normalization(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):

        return self.norm(x)
    
class FeedForward(nn.Module):

    def __init__(self,
                 hidden_size,
                 dropout=0.3):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):

        return self.net(x)        
    

# src/models/components.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):

        # x -> (batch, seq_len, hidden_size)

        weights = self.attn(x)

        weights = F.softmax(weights, dim=1)

        context = (weights * x).sum(dim=1)

        return context    