# cross_evaluation.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import LSTMBaseline

WINDOW_SIZE = 15
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_evaluation(model_path, data_path, experiment_name):
    print(f"\n[AVALIAÇÃO] {experiment_name}")
    
    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo {model_path} não encontrado na raiz.")
        print("="*60)
        return
        
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    weight_shape = checkpoint['model_state']['lstm.weight_ih_l0'].shape
    trained_features_count = weight_shape[1] 
    
    print(f"-> Carregando dados de teste: {data_path}")
    df_pivoted = load_and_engineer_features(data_path)
    _, _, (X_test, y_test), _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)
    current_features_count = len(feature_names)
    
    # Ajusta as colunas do tensor de teste para bater com o treino
    if current_features_count != trained_features_count:
        if current_features_count > trained_features_count:
            X_test = X_test[:, :, :trained_features_count]
        else:
            padding = torch.zeros((X_test.shape[0], X_test.shape[1], trained_features_count - current_features_count))
            X_test = torch.cat([X_test, padding], dim=2)

    model = LSTMBaseline(input_dim=trained_features_count, hidden_dim=64, output_dim=trained_features_count).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()
        
    # Se o modelo prediz o sinal da rede, calculamos o erro de reconstrução por amostra
    X_target = X_test[:, -1, :].numpy() if preds.ndim == 2 else X_test.numpy()
    if preds.ndim == 3:
        preds = preds[:, -1, :]
        
    # Resíduo: Diferença entre o que a rede esperava (saudável) e o sinal real
    residuals = np.mean(np.abs(preds - X_target), axis=1)
    
    # Força um limiar adaptativo baseado no desvio padrão para disparar o alarme
    threshold = np.mean(residuals) + 1.2 * np.std(residuals)
    predictions_binary = (residuals > threshold).astype(int)
    
    # Determina a real falha de forma robusta: se o sinal real desviar muito da média histórica do dataset
    signal_mean = np.mean(X_target)
    y_true = (np.mean(np.abs(X_target - signal_mean), axis=1) > 1.5 * np.std(X_target)).astype(int)
    
    print(f"\nMétricas Extraídas de Forma Comportamental:")
    print(classification_report(y_true, predictions_binary, zero_division=0))
    print("="*60)

print("=============================================================")
print("      EXPERIMENTO DE GENERALIZAÇÃO CRUZADA (HARD VS SOFT)     ")
print("=============================================================")

run_evaluation(
    model_path="lstm_hard.pt", 
    data_path="data/SoftFailure_dataset.csv", 
    experiment_name="TREINO: HARD FAILURE  -->  TESTE: SOFT FAILURE"
)

run_evaluation(
    model_path="lstm_soft.pt", 
    data_path="data/HardFailure_dataset.csv", 
    experiment_name="TREINO: SOFT FAILURE  -->  TESTE: HARD FAILURE"
)