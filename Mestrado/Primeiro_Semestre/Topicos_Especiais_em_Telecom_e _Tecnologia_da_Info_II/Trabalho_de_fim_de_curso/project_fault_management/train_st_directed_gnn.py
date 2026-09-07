# train_st_directed_gnn.py
import os
import torch
import torch.nn as nn
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import SpatioTemporalDirectedGNN
from src.evaluation import calculate_thresholds

WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 20
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando telemetria e gerando estrutura espacial de nós...")
df_pivoted = load_and_engineer_features(DATA_PATH)
train_loader, _, _, _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)
num_features = len(feature_names)

print(f"\n2. Inicializando ST-GNN Direcionada (Nós Monitorados: {num_features})...")
model = SpatioTemporalDirectedGNN(num_nodes=num_features, in_features=1, hidden_dim=32, output_dim=num_features).to(DEVICE)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("3. Iniciando treinamento focado em propagação causal...")
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
    print(f"Época [{epoch+1}/{EPOCHS}] - Erro de Propagação Causal: {train_loss:.4f}")

print("\n4. Calibrando limiares de detecção topológica...")
thresholds = calculate_thresholds(model, train_loader, DEVICE)

torch.save({
    'model_state': model.state_dict(),
    'thresholds': thresholds,
    'features': feature_names
}, "st_directed_gnn_baseline.pt")

print("[SUCESSO] Modelo Spatio-Temporal Directed GNN salvo em 'st_directed_gnn_baseline.pt'!")