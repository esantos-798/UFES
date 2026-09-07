# plot_classification_results.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import VanillaRNNBaseline, LSTMBaseline

# Configurações
WINDOW_SIZE = 15
BATCH_SIZE = 64
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INICIO_FALHA_TIMESTAMP = 1623416500

os.makedirs("plots", exist_ok=True)

print("1. Processando dados para o gráfico...")
df_pivoted = load_and_engineer_features(DATA_PATH)
_, _, (X_test, y_test), scaler, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

num_features = len(feature_names)
n_total = len(df_pivoted)
val_end = int(n_total * 0.80)
test_timestamps = df_pivoted['Timestamp'].values[val_end + WINDOW_SIZE:]

# Ground Truth
y_true = np.where(test_timestamps >= INICIO_FALHA_TIMESTAMP, 1, 0)

def get_predictions(model_path, model_class, name):
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = model_class(input_dim=num_features, hidden_dim=64, output_dim=num_features).to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    thresholds = checkpoint['thresholds']
    with torch.no_grad():
        preds = model(X_test.to(DEVICE)).cpu().numpy()
    
    errors = np.abs(y_test.numpy() - preds)
    y_pred = np.zeros(len(errors))
    for t in range(len(errors)):
        for f_idx in range(num_features):
            if errors[t, f_idx] > thresholds[f_idx]:
                y_pred[t] = 1
                break
    return y_pred

print("2. Extraindo predições dos modelos...")
y_pred_rnn = get_predictions("rnn_baseline.pt", VanillaRNNBaseline, "RNN")
y_pred_lstm = get_predictions("lstm_baseline.pt", LSTMBaseline, "LSTM")

print("3. Gerando gráfico comparativo de alarmes...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(14, 6))

# Plota as linhas horizontais deslocadas no eixo Y para não sobreporem totalmente
# Multiplicamos por pequenos fatores ou somamos offsets para criar degraus visuais discretos
ax.plot(test_timestamps, y_true, label="Ground Truth (Realidade)", color="black", linewidth=2.5, zorder=1)
ax.plot(test_timestamps, y_pred_rnn * 0.95 + 0.02, label="Alarmes Disparados - Vanilla RNN", color="crimson", alpha=0.8, linestyle="--", linewidth=1.5)
ax.plot(test_timestamps, y_pred_lstm * 0.90 + 0.04, label="Alarmes Disparados - LSTM Baseline", color="royalblue", alpha=0.8, linestyle=":", linewidth=2)

ax.set_title("Comparação Temporal de Detecção de Falhas (Hard Failure)", fontsize=14, fontweight='bold')
ax.set_xlabel("Timestamp", fontsize=12)
ax.set_ylabel("Estado do Alarme (0: Saudável | 1: Anomalia)", fontsize=12)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Normal (0)", "Falha (1)"])

# Destaca visualmente a região onde a falha foi de fato injetada no simulador
ax.axvspan(INICIO_FALHA_TIMESTAMP, test_timestamps[-1], color='red', alpha=0.07, label='Janela Real de Falha')

ax.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='gainsboro')
plt.tight_layout()

plot_out = "plots/alarm_classification_comparison.png"
plt.savefig(plot_out, dpi=300)
plt.close()

print(f"\n[SUCESSO] Gráfico de classificação gerado em: '{plot_out}'")