# detect_faults.py
import torch
import numpy as np
import pandas as pd
import os
from src.data_processing import load_and_engineer_features, prepare_datasets
# 1. Ajuste do import para trazer a LSTNet Modificada
from src.models import LSTNetModified
#from src.models import LSTMBaseline
#from src.models import VanillaRNNBaseline

# --- CONFIGURAÇÕES ---
WINDOW_SIZE = 15
BATCH_SIZE = 64
#DATA_PATH = "data/HardFailure_dataset.csv"
DATA_PATH = "data/SoftFailure_dataset.csv"
# 2. Caminho apontando para o arquivo gerado pela LSTNet
#MODEL_PATH = "lstnet_hard_failure.pt" 
#MODEL_PATH = "lstm_soft_failure_baseline.pt" 
#MODEL_PATH = "rnn_soft_failure_baseline.pt" 
MODEL_PATH = "lstnet_soft_failure.pt" 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando dados e modelo LSTNet salvo...")
df_pivoted = load_and_engineer_features(DATA_PATH)
_, _, (X_test, y_test), scaler, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

# Carrega os pesos e limiares contornando a restrição do PyTorch 2.6
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
thresholds = checkpoint['thresholds']

# 3. Inicialização correta da LSTNet Modificada
num_features = len(feature_names)
model = LSTNetModified(input_dim=num_features, window_size=WINDOW_SIZE, hidden_dim=64).to(DEVICE)
#model = LSTMBaseline(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
#model = VanillaRNNBaseline(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print("2. Executando predições no conjunto de teste (Cenário de Falha)...")
X_test = X_test.to(DEVICE)
with torch.no_grad():
    preds = model(X_test).cpu().numpy()

targets = y_test.numpy()

# 4. Cálculo dos Indicadores de Falha (Fault Indicators - FI)
test_errors = np.abs(targets - preds)

print("\n3. Analisando alarmes e detecção de anomalias com LSTNet...")
n_total = len(df_pivoted)
val_end = int(n_total * 0.80)
test_timestamps = df_pivoted['Timestamp'].values[val_end + WINDOW_SIZE:]

alarms_triggered = 0
total_test_points = len(test_errors)

print(f"Total de pontos temporais testados: {total_test_points}")

for t in range(total_test_points):
    fired_features = []
    for f_idx, feature_name in enumerate(feature_names):
        if test_errors[t, f_idx] > thresholds[f_idx]:
            fired_features.append(feature_name)
            
    if fired_features:
        alarms_triggered += 1
        if alarms_triggered <= 10: 
            print(f" [ALARME] Timestamp {test_timestamps[t]} -> Falha detectada em: {fired_features}")
        elif alarms_triggered == 11:
            print(" ... (outros alarmes omitidos para simplificar o log) ...")

print(f"\nAnálise concluída. Pontos com anomalia: {alarms_triggered} de {total_test_points} ({alarms_triggered/total_test_points*100:.2f}%)")