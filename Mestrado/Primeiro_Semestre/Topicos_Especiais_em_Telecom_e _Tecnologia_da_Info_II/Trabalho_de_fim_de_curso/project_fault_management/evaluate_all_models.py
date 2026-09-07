# evaluate_all_models.py
import os
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import VanillaRNNBaseline, LSTMBaseline, GRUBaseline, BiLSTMBaseline, LSTNetModified, AttentionLSTNet, TransformerBaseline, GraphTemporalFusion, SpatioTemporalDirectedGNN

# --- CONFIGURAÇÕES ---
WINDOW_SIZE = 15
BATCH_SIZE = 64
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INICIO_FALHA_TIMESTAMP = 1623416500
SMOOTHING_WINDOW = 5 # Janela de persistência para evitar alarmes piscantes

print("1. Preparando dados e gerando Ground Truth do cenário...")
df_pivoted = load_and_engineer_features(DATA_PATH)
_, _, (X_test, y_test), scaler, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

num_features = len(feature_names)
n_total = len(df_pivoted)
val_end = int(n_total * 0.80)
test_timestamps = df_pivoted['Timestamp'].values[val_end + WINDOW_SIZE:]

# Criação do vetor real de falhas
y_true = np.where(test_timestamps >= INICIO_FALHA_TIMESTAMP, 1, 0)

def apply_smoothing(y_pred_raw, window_size=SMOOTHING_WINDOW):
    """Aplica filtro de persistência: se houver ativação na janela, consolida o alarme"""
    y_smooth = np.copy(y_pred_raw)
    for i in range(window_size, len(y_pred_raw)):
        if np.sum(y_pred_raw[i-window_size:i+1]) >= 1:
            y_smooth[i] = 1
    return y_smooth

def eval_pytorch_model(model_path, model_class, name):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    if name in ["LSTNet", "Attention-LSTNet"]:
        model = model_class(input_dim=num_features, window_size=WINDOW_SIZE, hidden_dim=64).to(DEVICE)
    elif name == "Transformer":
        model = model_class(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
    elif name == "GCN+LSTM":
        model = model_class(num_nodes=num_features, in_features=1, hidden_dim=32, output_dim=num_features).to(DEVICE)
    elif name == "ST-GNN Direcionada":
        model = model_class(num_nodes=num_features, in_features=1, hidden_dim=32, output_dim=num_features).to(DEVICE)
    else:
        model = model_class(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
        
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    thresholds = checkpoint['thresholds']
    
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()
        
    errors = np.abs(y_test.numpy() - preds)
    y_pred_raw = np.zeros(len(errors))
    
    for t in range(len(errors)):
        for f_idx in range(num_features):
            if errors[t, f_idx] > thresholds[f_idx]:
                y_pred_raw[t] = 1
                break
                
    y_pred = apply_smoothing(y_pred_raw)
    return calculate_metrics(y_true, y_pred, name)

def eval_lstm_xgboost():
    name = "LSTM + XGBoost"
    if not os.path.exists("lstm_xgboost_pipeline.pkl"):
        return None
        
    with open("lstm_xgboost_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
        
    # Reconstruindo a LSTM para extração
    lstm_model = LSTMBaseline(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
    lstm_model.load_state_dict(pipeline['lstm_state'])
    lstm_model.eval()
    
    xgb_model = pipeline['xgb_model']
    thresholds = pipeline['thresholds']
    
    with torch.no_grad():
        lstm_out, _ = lstm_model.lstm(X_test.to(DEVICE))
        last_hidden = lstm_out[:, -1, :].cpu().numpy()
        
    preds = xgb_model.predict(last_hidden)
    errors = np.abs(y_test.numpy() - preds)
    y_pred_raw = np.zeros(len(errors))
    
    for t in range(len(errors)):
        for f_idx in range(num_features):
            if errors[t, f_idx] > thresholds[f_idx]:
                y_pred_raw[t] = 1
                break
                
    y_pred = apply_smoothing(y_pred_raw)
    return calculate_metrics(y_true, y_pred, name)

def calculate_metrics(y_t, y_p, name):
    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, zero_division=0)
    rec = recall_score(y_t, y_p, zero_division=0)
    f1 = f1_score(y_t, y_p, zero_division=0)
    return [name, acc, prec, rec, f1]

# --- EXECUÇÃO DOS TESTES ---
results = []

models_to_run = [
    ("rnn_baseline.pt", VanillaRNNBaseline, "Vanilla RNN"),
    ("lstm_baseline.pt", LSTMBaseline, "LSTM Baseline"),
    ("gru_baseline.pt", GRUBaseline, "GRU Baseline"),
    ("bilstm_baseline.pt", BiLSTMBaseline, "BiLSTM Baseline"),
    ("lstnet_hard_failure.pt", LSTNetModified, "LSTNet"),
    ("attention_lstnet.pt", AttentionLSTNet, "Attention-LSTNet"),
    ("transformer_baseline.pt", TransformerBaseline, "Transformer"),
    ("gcn_temporal_baseline.pt", GraphTemporalFusion, "GCN+LSTM"),
    ("st_directed_gnn_baseline.pt", SpatioTemporalDirectedGNN, "ST-GNN Direcionada")
]

for path, cls, name in models_to_run:
    if os.path.exists(path):
        results.append(eval_pytorch_model(path, cls, name))

xgb_res = eval_lstm_xgboost()
if xgb_res:
    results.append(xgb_res)

# Montando Dataframe final
df_res = pd.DataFrame(results, columns=["Modelo", "Accuracy", "Precision", "Recall", "F1-Score"])
print("\n" + "="*65)
print("       TABELA COMPARATIVA FINAL (MÉTRICAS DE CLASSIFICAÇÃO)")
print("="*65)
print(df_res.to_string(index=False, formatters={
    'Accuracy': '{:,.4f}'.format, 'Precision': '{:,.4f}'.format,
    'Recall': '{:,.4f}'.format, 'F1-Score': '{:,.4f}'.format
}))
print("="*65)