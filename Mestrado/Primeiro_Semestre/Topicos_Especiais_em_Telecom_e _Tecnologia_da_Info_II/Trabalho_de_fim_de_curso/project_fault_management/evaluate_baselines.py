# evaluate_baselines.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import VanillaRNNBaseline, LSTMBaseline

# --- CONFIGURAÇÕES ---
WINDOW_SIZE = 15
BATCH_SIZE = 64
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando dados e preparando conjuntos...")
df_pivoted = load_and_engineer_features(DATA_PATH)
_, _, (X_test, y_test), scaler, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

num_features = len(feature_names)
n_total = len(df_pivoted)
val_end = int(n_total * 0.80)
test_timestamps = df_pivoted['Timestamp'].values[val_end + WINDOW_SIZE:]

# --- DEFINIÇÃO DO GROUND TRUTH (RÓTULOS REAIS) ---
# Com base no gráfico da EDA, a falha começa visivelmente no timestamp ~1623416500.
# Vamos criar o vetor real: 1 para falha/anomalia, 0 para normal.
INICIO_FALHA_TIMESTAMP = 1623416500
y_true = np.where(test_timestamps >= INICIO_FALHA_TIMESTAMP, 1, 0)

def evaluate_model(model_path, model_class, name):
    print(f"\nAvaliando {name}...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Inicializa e carrega os pesos
    if name == "LSTNet":
        model = model_class(input_dim=num_features, window_size=WINDOW_SIZE, hidden_dim=64).to(DEVICE)
    else:
        model = model_class(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
        
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    thresholds = checkpoint['thresholds']
    
    # Predições
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()
    
    targets = y_test.numpy()
    errors = np.abs(targets - preds)
    
    # Se QUALQUER uma das features estourar o seu respectivo limiar, determinamos como anomalia (1)
    y_pred = np.zeros(len(errors))
    for t in range(len(errors)):
        for f_idx in range(num_features):
            if errors[t, f_idx] > thresholds[f_idx]:
                y_pred[t] = 1
                break # Já detectou anomalia nesse ponto, pula para o próximo timestamp
                
    # Cálculo das métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"[{name}] Accuracy : {acc:.4f}")
    print(f"[{name}] Precision: {prec:.4f}")
    print(f"[{name}] Recall   : {rec:.4f}")
    print(f"[{name}] F1-Score : {f1:.4f}")
    
    return [name, acc, prec, rec, f1]

results = []

# Avalia RNN
if os.path.exists("rnn_baseline.pt"):
    results.append(evaluate_model("rnn_baseline.pt", VanillaRNNBaseline, "Vanilla RNN"))

# Avalia LSTM
if os.path.exists("lstm_baseline.pt"):
    results.append(evaluate_model("lstm_baseline.pt", LSTMBaseline, "LSTM Baseline"))

# Consolida na tabela do terminal
df_res = pd.DataFrame(results, columns=["Modelo", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n=== TABELA COMPARATIVA DE CLASSIFICAÇÃO ===")
print(df_res.to_string(index=False))