# train_lstm_xgboost.py
import os
import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import LSTMBaseline
from src.evaluation import calculate_thresholds

# --- CONFIGURAÇÕES ---
WINDOW_SIZE = 15
BATCH_SIZE = 64
EPOCHS = 15
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando dados para o modelo híbrido LSTM+XGBoost...")
df_pivoted = load_and_engineer_features(DATA_PATH)
train_loader, val_loader, (X_test, y_test), scaler, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)
num_features = len(feature_names)

print("\n2. Etapa I: Treinando o extrator de características (LSTM)...")
lstm_model = LSTMBaseline(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    lstm_model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        preds = lstm_model(batch_X)
        loss = criterion(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- EXTRAÇÃO DE RECURSOS (HIDDEN STATES) ---
print("\n3. Etapa II: Extraindo representações latentes com a LSTM...")
lstm_model.eval()

def extract_hidden_states(loader):
    hidden_features = []
    targets = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            # Para extrair o comportamento temporal puro, capturamos a saída da camada LSTM
            # contornando a projeção linear final do forward padrão
            x_device = batch_X.to(DEVICE)
            lstm_out, _ = lstm_model.lstm(x_device)
            last_hidden = lstm_out[:, -1, :].cpu().numpy() # (Batch, hidden_dim)
            hidden_features.append(last_hidden)
            targets.append(batch_y.numpy())
    return np.vstack(hidden_features), np.vstack(targets)

X_train_xgb, y_train_xgb = extract_hidden_states(train_loader)

# --- TREINAMENTO DO XGBOOST REGRESSOR ---
print("\n4. Etapa III: Treinando o regressor XGBoost Multi-Output...")
# Como temos 15 features de saída, usamos o MultiOutputRegressor do sklearn acoplado ao XGBoost
from sklearn.multioutput import MultiOutputRegressor

xgb_core = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, tree_method="hist")
xgb_model = MultiOutputRegressor(xgb_core)
xgb_model.fit(X_train_xgb, y_train_xgb)

# --- CALIBRANDO LIMITES DE ERRO HÍBRIDOS ---
print("\n5. Calculando Limiares com base no resíduo do XGBoost...")
train_preds = xgb_model.predict(X_train_xgb)
residuals = np.abs(y_train_xgb - train_preds)
thresholds = np.percentile(residuals, 99, axis=0)

# Salva o pipeline completo (Pesos da LSTM + Modelo XGBoost via pickle + Limiares)
import pickle
output_data = {
    'lstm_state': lstm_model.state_dict(),
    'xgb_model': xgb_model,
    'thresholds': thresholds,
    'features': feature_names
}

with open("lstm_xgboost_pipeline.pkl", "wb") as f:
    pickle.dump(output_data, f)

print("\n[SUCESSO] Pipeline híbrido LSTM+XGBoost guardado em 'lstm_xgboost_pipeline.pkl'!")