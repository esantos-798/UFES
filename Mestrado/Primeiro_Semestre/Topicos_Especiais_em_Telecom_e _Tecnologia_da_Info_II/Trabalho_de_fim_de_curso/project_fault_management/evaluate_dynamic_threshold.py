# evaluate_dynamic_threshold.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import AttentionLSTNet, TransformerBaseline

# --- CONFIGURAÇÕES ---
WINDOW_SIZE = 15
BATCH_SIZE = 64
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INICIO_FALHA_TIMESTAMP = 1623416500

print("1. Preparando dados de telemetria...")
df_pivoted = load_and_engineer_features(DATA_PATH)
_, _, (X_test, y_test), _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

num_features = len(feature_names)
n_total = len(df_pivoted)
val_end = int(n_total * 0.80)
test_timestamps = df_pivoted['Timestamp'].values[val_end + WINDOW_SIZE:]

# Ground Truth real
y_true = np.where(test_timestamps >= INICIO_FALHA_TIMESTAMP, 1, 0)

def compute_ewma_threshold(errors, alpha=0.1, k=3.0, freeze_patience=2):
    """
    Calcula o limiar adaptativo dinâmico com congelamento retroativo isolado (t-2).
    Evita que a variância do primeiro impacto contamine o limiar operacional.
    """
    timesteps, features = errors.shape
    y_pred_alarm = np.zeros(timesteps)
    
    ewma_mean = np.copy(errors[0])
    ewma_var = np.zeros(features)
    
    # Histórico de limites calculados para recuperação segura
    history_thresholds = []
    
    consecutive_anomalies = 0
    is_frozen = False
    saved_threshold = None
    
    for t in range(1, timesteps):
        ewma_std = np.sqrt(ewma_var)
        calculated_threshold = ewma_mean + k * ewma_std
        
        # Alimenta o histórico com o limiar calculado deste ponto antes do choque
        history_thresholds.append(np.copy(calculated_threshold))
        if len(history_thresholds) > 5:
            history_thresholds.pop(0)
            
        # 1. Define qual limiar usar operacionalmente neste timestep
        if is_frozen and saved_threshold is not None:
            current_threshold = saved_threshold
        elif consecutive_anomalies > 0 and saved_threshold is not None:
            # Mesmo antes de consolidar o congelamento (durante a paciência),
            # usamos o backup para evitar que o limiar suba junto com o erro
            current_threshold = saved_threshold
        else:
            current_threshold = calculated_threshold
            
        # Atualiza as estatísticas apenas se não estivermos em estado de congelamento confirmado
        if not is_frozen:
            ewma_mean = alpha * errors[t] + (1 - alpha) * ewma_mean
            ewma_var = alpha * (errors[t] - ewma_mean)**2 + (1 - alpha) * ewma_var
            
        # 2. Avalia a violação do limiar adotado
        if np.any(errors[t] > current_threshold):
            y_pred_alarm[t] = 1
            consecutive_anomalies += 1
            
            # No exato primeiro passo de suspeita, recupera o limiar saudável de t-2
            if consecutive_anomalies == 1 and not is_frozen and len(history_thresholds) >= 2:
                saved_threshold = history_thresholds[-2]
                
            if consecutive_anomalies >= freeze_patience:
                is_frozen = True
        else:
            consecutive_anomalies = 0
            is_frozen = False
            saved_threshold = None
            
    return y_pred_alarm

def eval_with_dynamic_threshold(model_path, model_class, name):
    print(f" -> Processando resíduos dinâmicos para: {name}...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    if name == "Attention-LSTNet":
        model = model_class(input_dim=num_features, window_size=WINDOW_SIZE, hidden_dim=64).to(DEVICE)
    else:
        model = model_class(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
        
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()
        
    targets = y_test.numpy()
    errors = np.abs(targets - preds)
    
    # Aplica a nossa proposta de Limiar Dinâmico Adaptativo (sem precisar de suavização externa)
    y_pred_dynamic = compute_ewma_threshold(errors, alpha=0.1, k=2.5)
    
    acc = accuracy_score(y_true, y_pred_dynamic)
    prec = precision_score(y_true, y_pred_dynamic, zero_division=0)
    rec = recall_score(y_true, y_pred_dynamic, zero_division=0)
    f1 = f1_score(y_true, y_pred_dynamic, zero_division=0)
    
    return [name, acc, prec, rec, f1]

# --- EXECUÇÃO COMPARATIVA ---
dynamic_results = []

if os.path.exists("attention_lstnet.pt"):
    dynamic_results.append(eval_with_dynamic_threshold("attention_lstnet.pt", AttentionLSTNet, "Attention-LSTNet"))

if os.path.exists("transformer_baseline.pt"):
    dynamic_results.append(eval_with_dynamic_threshold("transformer_baseline.pt", TransformerBaseline, "Transformer"))

df_dyn = pd.DataFrame(dynamic_results, columns=["Modelo", "Accuracy", "Precision", "Recall", "F1-Score"])

print("\n" + "="*70)
print("   RESULTADO DAS SUAS MELHORIAS COM LIMIAR DINÂMICO ADAPTATIVO (EWMA)")
print("="*70)
print(df_dyn.to_string(index=False, formatters={
    'Accuracy': '{:,.4f}'.format, 'Precision': '{:,.4f}'.format,
    'Recall': '{:,.4f}'.format, 'F1-Score': '{:,.4f}'.format
}))
print("="*70)