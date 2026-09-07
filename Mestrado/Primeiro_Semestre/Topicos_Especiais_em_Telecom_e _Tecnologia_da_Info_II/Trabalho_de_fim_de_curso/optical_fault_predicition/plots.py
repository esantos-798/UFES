import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Estrutura de Dados Fornecida
# -----------------------------------------------------------------------------
data = [
    # HARD FAILURE
    {"dataset": "hard", "model": "lstm", "F1": 0.211454, "Best_F1": 0.883721, "Best_AUC": 0.997948, "lead_time_avg": 8.431818},
    {"dataset": "hard", "model": "tcn", "F1": 0.215054, "Best_F1": 0.873563, "Best_AUC": 0.997619, "lead_time_avg": 5.931818},
    {"dataset": "hard", "model": "bilstm", "F1": 0.226277, "Best_F1": 0.857143, "Best_AUC": 0.997929, "lead_time_avg": 6.090909},
    {"dataset": "hard", "model": "transformer", "F1": 0.186047, "Best_F1": 0.857143, "Best_AUC": 0.997716, "lead_time_avg": 4.613636},
    {"dataset": "hard", "model": "multitask_lstnet_tcn", "F1": 0.135849, "Best_F1": 0.847059, "Best_AUC": 0.996109, "lead_time_avg": 6.909091},
    {"dataset": "hard", "model": "gru", "F1": 0.173684, "Best_F1": 0.833333, "Best_AUC": 0.997503, "lead_time_avg": 4.318182},
    {"dataset": "hard", "model": "multitask_lstnet_attention", "F1": 0.125654, "Best_F1": 0.823529, "Best_AUC": 0.995606, "lead_time_avg": 3.954545},
    {"dataset": "hard", "model": "lstnet", "F1": 0.185567, "Best_F1": 0.809524, "Best_AUC": 0.996283, "lead_time_avg": 5.636364},
    {"dataset": "hard", "model": "multitask_lstnet_transformer", "F1": 0.157761, "Best_F1": 0.809524, "Best_AUC": 0.996167, "lead_time_avg": 4.022727},
    {"dataset": "hard", "model": "multitask_lstnet", "F1": 0.130952, "Best_F1": 0.804878, "Best_AUC": 0.996090, "lead_time_avg": 5.409091},
    {"dataset": "hard", "model": "attention_lstnet", "F1": 0.178490, "Best_F1": 0.790123, "Best_AUC": 0.995683, "lead_time_avg": 3.863636},
    
    # SOFT FAILURE
    {"dataset": "soft", "model": "lstm", "F1": 0.280220, "Best_F1": 0.937799, "Best_AUC": 0.974204, "lead_time_avg": 3.990566},
    {"dataset": "soft", "model": "bilstm", "F1": 0.270833, "Best_F1": 0.937799, "Best_AUC": 0.973313, "lead_time_avg": 3.952830},
    {"dataset": "soft", "model": "gru", "F1": 0.269720, "Best_F1": 0.937799, "Best_AUC": 0.975425, "lead_time_avg": 3.613208},
    {"dataset": "soft", "model": "tcn", "F1": 0.315522, "Best_F1": 0.937799, "Best_AUC": 0.977186, "lead_time_avg": 3.859813},
    {"dataset": "soft", "model": "transformer", "F1": 0.256303, "Best_F1": 0.937799, "Best_AUC": 0.972653, "lead_time_avg": 3.046729},
    {"dataset": "soft", "model": "lstnet", "F1": 0.251889, "Best_F1": 0.937799, "Best_AUC": 0.975665, "lead_time_avg": 3.688679},
    {"dataset": "soft", "model": "multitask_lstnet", "F1": 0.425000, "Best_F1": 0.937799, "Best_AUC": 0.977746, "lead_time_avg": 4.377358},
    {"dataset": "soft", "model": "multitask_lstnet_tcn", "F1": 0.329412, "Best_F1": 0.937799, "Best_AUC": 0.975295, "lead_time_avg": 4.490566},
    {"dataset": "soft", "model": "multitask_lstnet_transformer", "F1": 0.280460, "Best_F1": 0.937799, "Best_AUC": 0.977476, "lead_time_avg": 3.113208},
    {"dataset": "soft", "model": "multitask_lstnet_attention", "F1": 0.359477, "Best_F1": 0.932692, "Best_AUC": 0.975345, "lead_time_avg": 5.254717},
    {"dataset": "soft", "model": "attention_lstnet", "F1": 0.273782, "Best_F1": 0.932692, "Best_AUC": 0.977066, "lead_time_avg": 3.102804},
]

df = pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 2. Configurações Globais de Estilo
# -----------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

# -----------------------------------------------------------------------------
# Figura 1: Comparação de Best F1-Score
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=df,
    x='model',
    y='Best_F1',
    hue='dataset',
    palette={'hard': '#c93b3b', 'soft': '#4682b4'},
    edgecolor='black',
    linewidth=1.2,
    ax=ax
)

ax.set_title("1. Comparação de Desempenho de Detecção: Best F1-Score", fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel("Modelo / Arquitetura", fontsize=12, labelpad=10)
ax.set_ylabel("Best F1-Score", fontsize=12)
ax.set_ylim(0, 1.05)
plt.xticks(rotation=35, ha='right')

for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(
            f'{height:.2f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=9.0, fontweight='bold',
            xytext=(0, 3), textcoords='offset points'
        )

plt.legend(title='Dataset', loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('01_f1_score_comparison.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Figura 2: Antecedência Média de Detecção (Lead Time)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=df,
    x='model',
    y='lead_time_avg',
    hue='dataset',
    palette={'hard': '#6897bb', 'soft': '#365c7e'},
    edgecolor='black',
    linewidth=1.2,
    ax=ax
)

ax.set_title("2. Antecedência Média de Detecção de Falha (Lead Time Médio em Passos Temporais)", fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel("Modelo / Arquitetura", fontsize=12, labelpad=10)
ax.set_ylabel("Lead Time (Passos Temporais)", fontsize=12)
plt.xticks(rotation=35, ha='right')

for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(
            f'{height:.1f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=9.0,
            xytext=(0, 3), textcoords='offset points'
        )

plt.legend(title='Dataset', loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('02_lead_time_distribution.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Figura 3: Capacidade Discriminativa (AUC-ROC)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=df,
    x='model',
    y='Best_AUC',
    hue='dataset',
    palette={'hard': '#365c7e', 'soft': '#48a97c'},
    edgecolor='black',
    linewidth=1.2,
    ax=ax
)

ax.set_title("3. Capacidade Discriminativa Geral: Área Sob a Curva ROC (Best AUC-ROC)", fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel("Modelo / Arquitetura", fontsize=12, labelpad=10)
ax.set_ylabel("AUC-ROC", fontsize=12)

# Ajuste automático do eixo Y com corte inferior limpo
min_auc = df['Best_AUC'].min()
ax.set_ylim(max(0.0, min_auc - 0.02), 1.01)

plt.xticks(rotation=35, ha='right')

for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height) and height > 0:
        ax.annotate(
            f'{height:.3f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            fontsize=8.0, rotation=90,
            xytext=(0, 5), textcoords='offset points'
        )

plt.legend(title='Dataset', loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('03_auc_roc_accuracy.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------------
# Figura 4: Trade-off de Pareto
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7))

sns.scatterplot(
    data=df,
    x='lead_time_avg',
    y='Best_F1',
    hue='model',
    style='dataset',
    markers={'hard': 'o', 'soft': 'X'},
    s=140,
    ax=ax
)

ax.set_title("4. Análise do Trade-off de Pareto: Lead Time vs. Best F1-Score", fontsize=14, pad=15, fontweight='bold')
ax.set_xlabel("Lead Time Médio (Passos de Antecedência) → Maior é melhor", fontsize=12, labelpad=10)
ax.set_ylabel("Best F1-Score → Maior é melhor", fontsize=12)

# Rotulando pontos sem dependência de libs externas
for _, row in df.iterrows():
    ax.annotate(
        row['model'],
        (row['lead_time_avg'], row['Best_F1']),
        xytext=(5, 5), textcoords='offset points',
        fontsize=8.5, alpha=0.85
    )

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig('04_pareto_f1_vs_leadtime.png', dpi=300)
plt.close()

print("✅ Todos os 4 gráficos foram gerados diretamente a partir da lista fornecida!")