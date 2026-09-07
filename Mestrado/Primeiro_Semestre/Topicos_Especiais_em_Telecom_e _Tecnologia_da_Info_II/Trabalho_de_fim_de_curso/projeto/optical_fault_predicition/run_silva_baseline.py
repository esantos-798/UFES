import os
import sys

# Força o stdout do Python a utilizar UTF-8 para evitar erros no Windows CP1252
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 1. ARQUITETURA LSTM ORIGINAL (Silva et al.)
class LSTM_SilvaEtAl(nn.Module):
    def __init__(self, input_dim=12, output_horizon=12):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 100, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(100, 80, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(80, 60, batch_first=True)
        self.drop3 = nn.Dropout(0.2)
        self.lstm4 = nn.LSTM(60, 40, batch_first=True)
        self.drop4 = nn.Dropout(0.2)
        
        m = output_horizon
        self.fc1 = nn.Linear(40, 3 * m)
        self.fc2 = nn.Linear(3 * m, 2 * m)
        self.out = nn.Linear(2 * m, output_horizon)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out)
        out, _ = self.lstm3(out)
        out = self.drop3(out)
        out, _ = self.lstm4(out)
        out = self.drop4(out)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        return self.out(out)

# 2. FUNÇÃO DE TREINO COM EARLY STOPPING
def train_silva_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando treino do Silva et al. no dispositivo: {device}")
    
    # Hiperparâmetros de Silva et al.
    lr = 0.0001
    batch_size = 128
    epochs = 100
    
    # TODO: Certifique-se de substituir este bloco pelos dados/DataLoaders do seu projeto
    X_train = torch.randn(1000, 10, 12)
    y_train = torch.randn(1000, 12)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTM_SilvaEtAl(input_dim=12, output_horizon=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoca [{epoch}/{epochs}] - Loss MSE (Z-Score): {avg_loss:.6f}")
        
    print("Treinamento concluido com sucesso!")

if __name__ == "__main__":
    train_silva_baseline()