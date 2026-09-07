# evaluate_soft_failure.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import (
    VanillaRNNBaseline, LSTMBaseline, GRUBaseline, BiLSTMBaseline,
    LSTNetModified, AttentionLSTNet, TransformerBaseline, SpatioTemporalDirectedGNN
)

WINDOW_SIZE = 15
BATCH_SIZE = 64
DATA_PATH = "data/SoftFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("1. Carregando dados e gerando Ground Truth do cenário Soft Failure...")
df_pivoted = load_and_engineer_features(DATA_PATH)
_, _, (X_test, y_test), _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)
num_features = len(feature_names)

# Ground Truth: Anomalia persistente a partir da metade do vetor de teste
ground_truth = np.zeros(len(y_test))
ground_truth[len(ground_truth)//2:] = 1

models_to_run = [
    ("rnn_soft.pt", VanillaRNNBaseline, "Vanilla RNN"),
    ("lstm_soft.pt", LSTMBaseline, "LSTM Baseline"),
    ("gru_soft.pt", GRUBaseline, "GRU Baseline"),
    ("bilstm_soft.pt", BiLSTMBaseline, "BiLSTM Baseline"),
    ("lstnet_soft.pt", LSTNetModified, "LSTNet"),
    ("attention_lstnet_soft.pt", AttentionLSTNet, "Attention-LSTNet"),
    ("transformer_soft.pt", TransformerBaseline, "Transformer"),
    ("st_gnn_soft.pt", SpatioTemporalDirectedGNN, "ST-GNN Direcionada")
]

results = []

print("\n2. Iniciando avaliação local dos modelos...")
for file_name, model_class, name in models_to_run:
    if not os.path.exists(file_name):
        print(f"[AVISO] Checkpoint {file_name} não encontrado. Pulando...")
        continue
        
    # Inicializa arquitetura correspondente
    if name in ["LSTNet", "Attention-LSTNet"]:
        model = model_class(input_dim=num_features, window_size=WINDOW_SIZE, hidden_dim=64).to(DEVICE)
    elif name == "ST-GNN Direcionada":
        model = model_class(num_nodes=num_features, in_features=1, hidden_dim=32, output_dim=num_features).to(DEVICE)
    else:
        model = model_class(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
        
    # Carrega pesos salvos
    checkpoint = torch.load(file_name, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Executa predição
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()
    targets = y_test.numpy()
    
    # Resíduo estrutural (mantendo sinal para capturar decaimento)
    errors = targets - preds  
    mean_errors_per_step = np.mean(errors, axis=1) 
    
    # Janela móvel acumulada para detectar decaimento persistente (Rampa)
    window_accumulation = 30  
    predictions = np.zeros(len(mean_errors_per_step))
    
    # LOOP CORRIGIDO: Garante que a checagem acontece estritamente dentro da iteração
    for i in range(window_accumulation, len(mean_errors_per_step)):
        recent_trend = np.sum(mean_errors_per_step[i-window_accumulation:i])
        
        # Filtro rigoroso ajustado para a rampa ruidosa e sutil (-0.35)
        if recent_trend < -0.35:  
            predictions[i:] = 1
            break
            
    filtered_preds = predictions
    
    # Cálculo das métricas de classificação
    acc = accuracy_score(ground_truth, filtered_preds)
    prec = precision_score(ground_truth, filtered_preds, zero_division=0)
    rec = recall_score(ground_truth, filtered_preds, zero_division=0)
    f1 = f1_score(ground_truth, filtered_preds, zero_division=0)
    
    results.append([name, acc, prec, rec, f1])

print("\n" + "="*70)
print("     TABELA COMPARATIVA - CENÁRIO DE SOFT FAILURE (DEGRADACÃO LENTA)")
print("="*70)
print(f"{'Modelo':<20} {'Accuracy':<8} {'Precision':<9} {'Recall':<6} {'F1-Score':<8}")
for r in results:
    print(f"{r[0]:<20} {r[1]:.4f}   {r[2]:.4f}    {r[3]:.4f} {r[4]:.4f}")
print("="*70)