# exploratory_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_processing import load_and_engineer_features

# Garante que a pasta para salvar os gráficos exista
os.makedirs("plots", exist_ok=True)

DATA_PATH = "data/HardFailure_dataset.csv"

print("1. Carregando e pivotando o dataset de Hard Failure...")
# Usando a função do seu pipeline para garantir o pivot correto (uma linha por timestamp)
df_pivoted = load_and_engineer_features(DATA_PATH)

print("\n2. Executando Análise Estatística...")
# -- Quantidade de timestamps
total_timestamps = len(df_pivoted)
print(f" -> Quantidade total de timestamps únicos: {total_timestamps}")

# -- Valores faltantes
missing_values = df_pivoted.isnull().sum().sum()
print(f" -> Quantidade total de valores faltantes (NaN): {missing_values}")

# -- Distribuição das Falhas (Exemplo com base no percentil ou comportamento do dataset)
# Como o dataset de treino/validação representa o estado saudável e o teste a falha,
# vamos analisar o resumo estatístico das principais colunas para entender a dispersão.
print("\nResumo estatístico das variáveis cruciais:")
cols_to_show = [c for c in df_pivoted.columns if 'Power' in c or 'OSNR' in c or 'BER' in c][:6]
print(df_pivoted[cols_to_show].describe().to_string())

print("\n3. Gerando Gráficos das Métricas Ópticas...")

# Configuração global dos gráficos para o relatório do seu TCC
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# Encontra colunas que representam cada métrica para plotar uma amostra representativa
col_ber = [c for c in df_pivoted.columns if 'BER' in c][0]
col_osnr = [c for c in df_pivoted.columns if 'OSNR' in c][0]
col_input = [c for c in df_pivoted.columns if 'InputPower' in c][0]
col_output = [c for c in df_pivoted.columns if 'OutputPower' in c][0]

# Plot de BER
axes[0].plot(df_pivoted['Timestamp'], df_pivoted[col_ber], color='darkred', label=col_ber, alpha=0.8)
axes[0].set_title("Evolução Temporal da Taxa de Erro de Bits (BER)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("BER")
axes[0].legend(loc="upper left")

# Plot de OSNR
axes[1].plot(df_pivoted['Timestamp'], df_pivoted[col_osnr], color='darkblue', label=col_osnr, alpha=0.8)
axes[1].set_title("Relação Sinal-Ruído Óptica (OSNR)", fontsize=12, fontweight='bold')
axes[1].set_ylabel("OSNR (dB)")
axes[1].legend(loc="upper left")

# Plot de InputPower
axes[2].plot(df_pivoted['Timestamp'], df_pivoted[col_input], color='darkgreen', label=col_input, alpha=0.8)
axes[2].set_title("Potência de Entrada (Input Power)", fontsize=12, fontweight='bold')
axes[2].set_ylabel("Potência (dBm)")
axes[2].legend(loc="upper left")

# Plot de OutputPower
axes[3].plot(df_pivoted['Timestamp'], df_pivoted[col_output], color='darkorange', label=col_output, alpha=0.8)
axes[3].set_title("Potência de Saída (Output Power)", fontsize=12, fontweight='bold')
axes[3].set_ylabel("Potência (dBm)")
axes[3].set_xlabel("Timestamp")
axes[3].legend(loc="upper left")

plt.tight_layout()
plot_path = "plots/optical_metrics_behavior.png"
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"\n[SUCESSO] Gráfico unificado salvo com alta resolução em: '{plot_path}'")