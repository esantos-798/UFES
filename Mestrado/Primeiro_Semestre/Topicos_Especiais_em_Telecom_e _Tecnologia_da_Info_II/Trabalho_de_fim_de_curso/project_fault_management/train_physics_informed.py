# train_physics_informed.py
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models import LSTMBaseline
from sklearn.metrics import classification_report

WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = "data/SoftFailure_PhysicsInformed.csv"

df = pd.read_csv(DATA_PATH)

# Localiza a label injetada ou original
label_col = 'Label_Det' if 'Label_Det' in df.columns else [c for c in df.columns if 'label' in c.lower()][0]
y_det = df[label_col].values

ignore_cols = ['Timestamp', 'index', 'Unnamed: 0'] + [c for c in df.columns if 'label' in c.lower() or 'failure' in c.lower() or 'sev' in c.lower() or 'loc' in c.lower()]
feature_cols = [c for c in df.columns if c not in ignore_cols]
num_features = len(feature_cols)

X_data = df[feature_cols].values

X_windows, y_targets, y_reg_targets = [], [], []
for i in range(len(df) - WINDOW_SIZE - 1):
    X_windows.append(X_data[i:i+WINDOW_SIZE])
    y_targets.append(y_det[i+WINDOW_SIZE])
    y_reg_targets.append(X_data[i+WINDOW_SIZE])

X_tensor = torch.tensor(np.array(X_windows), dtype=torch.float32)
y_reg_tensor = torch.tensor(np.array(y_reg_targets), dtype=torch.float32)
y_labels = np.array(y_targets)

# Divisão sequencial (Holdout) para preservar a rampa temporal de teste no final do arquivo
split_idx = int(0.75 * len(X_tensor))

X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train_reg, y_test_reg = y_reg_tensor[:split_idx], y_reg_tensor[split_idx:]
y_train_lbl, y_test_lbl = y_labels[:split_idx], y_labels[split_idx:]

train_dataset = TensorDataset(X_train, y_train_reg, torch.tensor(y_train_lbl, dtype=torch.float32))
test_dataset = TensorDataset(X_test, y_test_reg, torch.tensor(y_test_lbl, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # Mantém ordem temporal
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMBaseline(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.MSELoss()

print("[TREINO] Rodando épocas guiadas por física...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y, _ in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"-> Época {epoch+1:02d}/{EPOCHS} | MSE Loss: {epoch_loss/len(train_loader):.6f}")

model.eval()
all_residuals = []
with torch.no_grad():
    for batch_X, batch_y, _ in test_loader:
        preds = model(batch_X.to(DEVICE)).cpu().numpy()
        targets = batch_y.numpy()
        res = np.mean((preds - targets) ** 2, axis=1)
        all_residuals.extend(res)

all_residuals = np.array(all_residuals)
threshold = np.mean(all_residuals) + 0.8 * np.std(all_residuals)
preds_binary = (all_residuals > threshold).astype(int)

print("\n" + "="*60)
print("   RELATÓRIO METODOLÓGICO: PIPELINE COMPORTAMENTAL ÓPTICO")
print("="*60)
print(classification_report(y_test_lbl, preds_binary, zero_division=0))
print("="*60)

torch.save({'model_state': model.state_dict(), 'threshold': threshold, 'feature_names': feature_cols}, "lstm_physics_soft.pt")
print("[SALVO] Checkpoint 'lstm_physics_soft.pt' pronto.")