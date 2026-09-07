# train_lstnet.py
import torch
import torch.nn as nn
import os
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import LSTNetModified
from src.evaluation import calculate_thresholds

WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 30  # LSTNet pode precisar de um pouco mais de épocas para convergir os componentes
DATA_PATH = "data/SoftFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando dados para LSTNet (Hard Failure)...")
df_pivoted = load_and_engineer_features(DATA_PATH)
train_loader, val_loader, test_data, scaler, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

num_features = len(feature_names)

print("2. Inicializando a LSTNet Modificada...")
model = LSTNetModified(input_dim=num_features, window_size=WINDOW_SIZE, hidden_dim=64).to(DEVICE)
criterion = nn.L1Loss() # MAE absoluto robusto
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("3. Treinando a LSTNet...")
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

print("4. Calibrando Limiares com LSTNet...")
thresholds = calculate_thresholds(model, train_loader, DEVICE)

output_model = "lstnet_soft_failure.pt"
torch.save({
    'model_state': model.state_dict(),
    'thresholds': thresholds,
    'features': feature_names
}, output_model)

print(f"\n[SUCESSO] LSTNet salva em '{output_model}'!")