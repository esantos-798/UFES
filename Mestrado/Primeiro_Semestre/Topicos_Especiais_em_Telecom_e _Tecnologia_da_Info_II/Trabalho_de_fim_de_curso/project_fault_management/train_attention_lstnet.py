# train_attention_lstnet.py
import os
import torch
import torch.nn as nn
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import AttentionLSTNet
from src.evaluation import calculate_thresholds

WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 20
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Inicializando dados para a nova Attention-LSTNet...")
df_pivoted = load_and_engineer_features(DATA_PATH)
train_loader, val_loader, _, _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)
num_features = len(feature_names)

print("\n2. Treinando o modelo proposto com Atenção...")
model = AttentionLSTNet(input_dim=num_features, window_size=WINDOW_SIZE, hidden_dim=64).to(DEVICE)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
        
    train_loss /= len(train_loader.dataset)
    print(f"Época [{epoch+1}/{EPOCHS}] - Perda MAE: {train_loss:.4f}")

print("\n3. Calibrando limiares estatísticos (Percentil 99%)...")
thresholds = calculate_thresholds(model, train_loader, DEVICE)

torch.save({
    'model_state': model.state_dict(),
    'thresholds': thresholds,
    'features': feature_names
}, "attention_lstnet.pt")

print("[SUCESSO] Modelo de Atencao salvo em 'attention_lstnet.pt'!")