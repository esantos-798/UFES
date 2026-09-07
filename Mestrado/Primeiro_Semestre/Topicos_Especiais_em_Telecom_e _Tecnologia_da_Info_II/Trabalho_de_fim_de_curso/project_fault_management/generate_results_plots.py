# generate_results_plots.py
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import load_and_engineer_features, prepare_datasets
from src.models import (
    VanillaRNNBaseline, LSTMBaseline, GRUBaseline, 
    BiLSTMBaseline, LSTNetModified, AttentionLSTNet, 
    TransformerBaseline, GraphTemporalFusion
)

# Configurações de layout para a escrita (padrão IEEE/SBC)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13})
os.makedirs("plots", exist_ok=True)

WINDOW_SIZE = 15
BATCH_SIZE = 64
DATA_PATH = "data/HardFailure_dataset.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Dados obtidos no evaluate_all_models.py (Consolidado)
# Atualize este bloco dentro de generate_results_plots.py
data_metrics = {
    'Modelo': ['Vanilla RNN', 'LSTM Baseline', 'GRU Baseline', 'BiLSTM Baseline', 
               'LSTNet', 'Attention-LSTNet', 'Transformer', 'GCN+LSTM', 'ST-GNN Direcionada', 'LSTM + XGBoost'],
    'F1-Score': [0.9290, 0.9323, 0.9607, 0.9445, 0.9250, 0.9536, 0.9636, 0.9366, 0.9463, 0.9996],
    'Recall': [0.8674, 0.8732, 0.9243, 0.8948, 0.8605, 0.9114, 0.9297, 0.8807, 0.8980, 0.9993]
}
df_metrics = pd.DataFrame(data_metrics)

# --- GRÁFICO 1: COMPARATIVO DE MÉTRICAS (Mestrado/TCC Style) ---
print("Gerando Gráfico 1: Comparativo de Métricas...")
df_melted = pd.melt(df_metrics, id_vars=['Modelo'], value_vars=['F1-Score', 'Recall'], 
                    var_name='Métrica', value_name='Valor')

plt.figure(figsize=(11, 5))
ax = sns.barplot(x='Modelo', y='Valor', hue='Métrica', data=df_melted, palette='muted')
plt.ylim(0.80, 1.02)
plt.title("Comparativo Global de Desempenho no Cenário de Hard Failure")
plt.xlabel("Arquiteturas Avaliadas")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plots/comparativo_metricas_tcc.png", dpi=300)
plt.close()

# --- GRÁFICO 2: COMPORTAMENTO DO RESÍDUO (ANÁLISE DO LIMIAR) ---
print("Gerando Gráfico 2: Evolução Temporal dos Resíduos...")
df_pivoted = load_and_engineer_features(DATA_PATH)
_, _, (X_test, y_test), _, feature_names = prepare_datasets(df_pivoted, WINDOW_SIZE, BATCH_SIZE)

# Vamos carregar o Transformer como exemplo de análise de resíduo temporal
checkpoint = torch.load("transformer_baseline.pt", map_location=DEVICE, weights_only=False)
model = TransformerBaseline(input_dim=len(feature_names), hidden_dim=64, output_dim=len(feature_names)).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

with torch.no_grad():
    preds = model(X_test.to(DEVICE)).cpu().numpy()
targets = y_test.numpy()
errors = np.abs(targets - preds)

# Seleciona o primeiro canal físico para plotar a evolução temporal do erro
canal_idx = 0
erro_canal = errors[:, canal_idx]
limiar_estatico = checkpoint['thresholds'][canal_idx]

plt.figure(figsize=(12, 4.5))
plt.plot(erro_canal, label=f'Resíduo de Predição (Canal {feature_names[canal_idx]})', color='royalblue', alpha=0.8)
plt.axhline(y=limiar_estatico, color='crimson', linestyle='--', linewidth=1.5, label='Limiar Estático (Percentil 99%)')

# Destaca visualmente o ponto onde ocorre o Hard Failure no conjunto de testes
# Ajuste o índice aproximado se a quebra acontecer antes ou depois no seu vetor de teste
plt.axvline(x=len(erro_canal)//2, color='darkorange', linestyle=':', label='Instante do Hard Failure (WSS Drop)')

plt.title("Análise Temporal do Erro de Predição vs Limiar de Decisão")
plt.xlabel("Timestamps (Conjunto de Teste)")
plt.ylabel("Erro Absoluto (MAE)")
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("plots/analise_residuo_temporal.png", dpi=300)
plt.close()

print("[SUCESSO] Gráficos gerados com qualidade de publicação na pasta 'plots/'!")