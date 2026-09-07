# train_rnn_baseline.py
import torch
import torch.nn as nn
import os
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import VanillaRNNBaseline
from src.evaluation import calculate_thresholds

# --- CONFIGURAÇÕES ---
WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 20
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando e processando dados para RNN (Hard Failure)...")
df_pivoted = load_and_engineer_features(DATA_PATH)
train_loader, val_loader, test_data, scaler, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

num_features = len(feature_names)
print(f"Número de features detectadas: {num_features}")

print("\n2. Inicializando o modelo Vanilla RNN Baseline...")
model = VanillaRNNBaseline(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
criterion = nn.L1Loss()  # MAE absoluto robusto conforme o artigo
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n3. Iniciando o Treinamento da RNN...")
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
    print(f"Época [{epoch+1}/{EPOCHS}] - Perda em Treino (MAE): {train_loss:.4f}")

print("\n4. Calculando os Limiares de Detecção (Percentil 99%)...")
thresholds = calculate_thresholds(model, train_loader, DEVICE)

# Salva os pesos e os limites específicos da RNN
output_model = "rnn_baseline.pt"
torch.save({
    'model_state': model.state_dict(),
    'thresholds': thresholds,
    'features': feature_names
}, output_model)

print(f"\n[SUCESSO] Modelo RNN e limiares guardados em '{output_model}'")