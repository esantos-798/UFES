# train_baseline.py
import torch
import torch.nn as nn
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import LSTMBaseline
from src.evaluation import calculate_thresholds

# Configurações
WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando e processando dados...")
df = load_and_engineer_features("data/HardFailure_dataset.csv")
train_loader, val_loader, test_data, scaler, feature_names = prepare_datasets(df, WINDOW_SIZE, BATCH_SIZE)

num_features = train_loader.dataset.tensors[0].shape[2]

print("2. Inicializando o modelo LSTM Baseline...")
model = LSTMBaseline(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
criterion = nn.L1Loss() # MAE conforme artigo
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("3. Iniciando o Treinamento...")
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
    print(f"Época [{epoch+1}/{EPOCHS}] - Perda: {train_loss:.4f}")

print("4. Calculando os Limiares de Detecção (Percentil 99%)...")
thresholds = calculate_thresholds(model, train_loader, DEVICE)
print("Limiares por feature calculados com sucesso!")

# Salva o modelo para usar no script de detecção posterior
torch.save({'model_state': model.state_dict(), 'thresholds': thresholds}, "lstm_baseline.pt")
print("Modelo salvo em 'lstm_baseline.pt'")