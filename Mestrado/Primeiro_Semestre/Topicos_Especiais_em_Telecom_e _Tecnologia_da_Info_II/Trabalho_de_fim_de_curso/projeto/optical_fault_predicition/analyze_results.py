import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CONFIGURAÇÕES E ESTILOS DE PLOTAGEM
# ==========================================================
RESULTS_DIR = "results"
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
FIGURES_DIR = os.path.join(SUMMARY_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Ordem de exibição dos modelos
MODEL_ORDER = [
    "lstm", "bilstm", "gru", "tcn", "transformer", "lstnet",
    "multitask_lstnet", "multitask_lstnet_attention", 
    "multitask_lstnet_tcn", "multitask_lstnet_transformer", 
    "attention_lstnet"
]

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15
})


# ==========================================================
# TRATAMENTO E CARREGAMENTO DE DADOS
# ==========================================================
def load_and_fix_dataset(csv_filename="summary.csv"):
    """
    Carrega o dataset e corrige anomalias:
    1. Trata 'model' como NaN preenchendo a partir do nome do experimento.
    2. Garante Best_F1 e Best_AUC pegando o MÁXIMO real entre Rede Pura e XGBoost.
    """
    csv_path = csv_filename
    if not os.path.exists(csv_path):
        found = glob.glob("**/summary.csv", recursive=True)
        if found:
            csv_path = found[0]
        else:
            raise FileNotFoundError(f"⚠️ O arquivo {csv_filename} não foi encontrado!")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Preenchimento de modelo faltante (ex: attention_lstnet)
    def infer_model(row):
        m = str(row.get("model", ""))
        if m != "nan" and m.strip() != "":
            return m.strip().lower()
        
        exp = str(row.get("experiment", ""))
        for token in ["_hard_", "_soft_"]:
            if token in exp:
                return exp.split(token)[0].lower()
        return exp.lower()

    df["model"] = df.apply(infer_model, axis=1)

    # Correção crítica de F1 e AUC (evita zeramento quando F1_XGB == 0.0)
    # Tolerante à ausência das colunas _XGB (ex: quando o módulo híbrido está desativado)
    def best_of(df, base_col, xgb_col):
        if xgb_col in df.columns:
            return df[[base_col, xgb_col]].max(axis=1)
        return df[base_col]

    df["Best_F1"] = best_of(df, "F1", "F1_XGB")
    df["Best_AUC"] = best_of(df, "AUC", "AUC_XGB")
    df["Best_Precision"] = best_of(df, "Precision", "Precision_XGB")
    df["Best_Recall"] = best_of(df, "Recall", "Recall_XGB")

    # Normalização de nomes de colunas de Lead Time se necessário
    if "Lead Time Avg" in df.columns and "lead_time_avg" not in df.columns:
        df["lead_time_avg"] = df["Lead Time Avg"]

    # Ordenação categórica dos modelos
    existing_models = [m for m in df["model"].unique() if pd.notna(m)]
    ordered_cats = [m for m in MODEL_ORDER if m in existing_models] + [m for m in existing_models if m not in MODEL_ORDER]
    df["model"] = pd.Categorical(df["model"], categories=ordered_cats, ordered=True)

    return df


# ==========================================================
# SUÍTE DE FIGURAS PARA O RELATÓRIO
# ==========================================================
def plot_f1_comparison(df, output_dir=FIGURES_DIR):
    """1. Comparação Geral de Best F1-Score"""
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x="model", y="Best_F1", hue="dataset", ci=None, edgecolor="black")
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, xytext=(0, 3),
                        textcoords='offset points')

    plt.title("Desempenho Geral de Detecção: Best F1-Score por Arquitetura")
    plt.xlabel("Arquitetura Neural")
    plt.ylabel("F1-Score Máximo (Rede vs XGBoost)")
    plt.xticks(rotation=40, ha="right")
    plt.ylim(0, 1.15)
    plt.legend(title="Dataset", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_f1_score_comparison.png"), dpi=300)
    plt.close()


def plot_forecast_metrics(df, output_dir=FIGURES_DIR):
    """2. Métricas da Etapa de Regressão/Forecast (MSE e R2)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    if "Forecast_MSE" in df.columns:
        sns.barplot(data=df, x="model", y="Forecast_MSE", hue="dataset", ax=axes[0], ci=None, edgecolor="black")
        axes[0].set_title("Forecast MSE (Menor é Melhor)")
        axes[0].set_xlabel("Modelo")
        axes[0].set_ylabel("MSE")
        axes[0].tick_params(axis='x', rotation=40)

    if "Forecast_R2" in df.columns:
        sns.barplot(data=df, x="model", y="Forecast_R2", hue="dataset", ax=axes[1], ci=None, edgecolor="black")
        axes[1].set_title("Forecast $R^2$ (Maior é Melhor)")
        axes[1].set_xlabel("Modelo")
        axes[1].set_ylabel("$R^2$")
        axes[1].set_ylim(-0.1, 1.05)
        axes[1].tick_params(axis='x', rotation=40)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_forecast_metrics.png"), dpi=300)
    plt.close()


def plot_lead_time_distribution(df, output_dir=FIGURES_DIR):
    """3. Antecedência Média de Detecção (Lead Time)"""
    if "lead_time_avg" not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="model", y="lead_time_avg", hue="dataset", ci=None, edgecolor="black", palette="Set2")

    plt.title("Antecedência Média de Detecção de Falha (Lead Time Médio em Passos Temporais)")
    plt.xlabel("Modelo")
    plt.ylabel("Lead Time (Passos)")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_lead_time_distribution.png"), dpi=300)
    plt.close()


def plot_pearson_vs_f1(df, output_dir=FIGURES_DIR):
    """4. Qualidade do Forecast (Pearson) vs Detecção (F1-Score)"""
    pearson_col = [c for c in df.columns if "pearson" in c.lower()]
    if not pearson_col:
        return

    col_name = pearson_col[0]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=col_name,
        y="Best_F1",
        hue="model",
        style="dataset",
        s=140,
        alpha=0.9
    )

    plt.title("Dispersão: Correlação Pearson do Forecast vs Best F1-Score")
    plt.xlabel("Pearson Correlation (Forecast)")
    plt.ylabel("Best F1-Score")
    plt.ylim(-0.05, 1.08)
    plt.xlim(-0.1, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_pearson_vs_f1.png"), dpi=300)
    plt.close()


def plot_auc_roc_summary(df, output_dir=FIGURES_DIR):
    """5. Área Sob a Curva ROC (AUC-ROC) Máxima"""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="model", y="Best_AUC", hue="dataset", ci=None, palette="viridis", edgecolor="black")
    plt.title("Capacidade Discriminativa de Falha: Área Sob a Curva ROC (Best AUC-ROC)")
    plt.xlabel("Modelo")
    plt.ylabel("AUC-ROC")
    plt.xticks(rotation=40, ha="right")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_auc_roc_summary.png"), dpi=300)
    plt.close()


# ==========================================================
# EXECUÇÃO PRINCIPAL E IMPRESSÃO DE TABELAS
# ==========================================================
def main():
    print("🚀 Iniciando processamento dos resultados dos experimentos...")
    
    # 1. Carregamento e limpeza completa
    df = load_and_fix_dataset()

    # Save cleaned consolidated table
    summary_csv_path = os.path.join(SUMMARY_DIR, "consolidated_results_cleaned.csv")
    df.to_csv(summary_csv_path, index=False)
    print(f"💾 Tabela limpa e consolidada salva em: {summary_csv_path}")

    # 2. Geração dos gráficos para o relatório
    print("🎨 Gerando as 5 figuras para o relatório em results/summary/figures/...")
    plot_f1_comparison(df)
    plot_forecast_metrics(df)
    plot_lead_time_distribution(df)
    plot_pearson_vs_f1(df)
    plot_auc_roc_summary(df)
    print("✅ Todas as figuras foram geradas com sucesso!")

    # 3. Tabela Comparativa dos Melhores Modelos no Terminal
    print("\n" + "="*85)
    print("🏆 MELHORES RESULTADOS POR MODELO E DATASET (ORDENADOS POR BEST_F1)")
    print("="*85)

    idx_best = df.groupby(["dataset", "model"], observed=True)["Best_F1"].idxmax().dropna()
    best_per_ds = df.loc[idx_best].sort_values(by=["dataset", "Best_F1"], ascending=[True, False])

    cols_to_print = [
        c for c in ["dataset", "model", "F1", "F1_XGB", "Best_F1", "Best_AUC", "Forecast_MSE", "lead_time_avg"] 
        if c in df.columns
    ]
    
    print(best_per_ds[cols_to_print].to_string(index=False))
    print("="*85 + "\n")


if __name__ == "__main__":
    main()