import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 1. CONFIGURAÇÕES E ESTILOS VISUAIS
# ==========================================================
RESULTS_DIR = "results"
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
FIGURES_DIR = os.path.join(SUMMARY_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Ordem de exibição dos modelos (apenas redes neurais)
MODEL_ORDER = [
    "lstm", "bilstm", "gru", "tcn", "transformer", "lstnet",
    "multitask_lstnet", "multitask_lstnet_attention", 
    "multitask_lstnet_tcn", "multitask_lstnet_transformer", 
    "attention_lstnet"
]

sns.set_theme(style="whitegrid", palette="deep")
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
# 2. CARREGAMENTO E FILTRAGEM DOS DADOS
# ==========================================================
def load_and_clean_dl_data(csv_filename="summary.csv"):
    """
    Carrega o dataset e filtra estritamente para modelos de Redes Neurais Puras,
    descartando experimentos com XGBoost.
    """
    csv_path = csv_filename
    if not os.path.exists(csv_path):
        found = glob.glob("**/summary.csv", recursive=True)
        if found:
            csv_path = found[0]
        else:
            raise FileNotFoundError(f"⚠️ Arquivo {csv_filename} não encontrado!")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Preenchimento e inferência do nome do modelo
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

    # REMOÇÃO ESTRITA DE MODELOS XGBOOST
    df = df[~df["model"].str.contains("xgboost", case=False, na=False)].copy()

    # Garantir conversão numérica das métricas puras da rede
    df["F1"] = pd.to_numeric(df["F1"], errors="coerce")
    df["AUC"] = pd.to_numeric(df["AUC"], errors="coerce")
    
    if "Precision" in df.columns:
        df["Precision"] = pd.to_numeric(df["Precision"], errors="coerce")
    if "Recall" in df.columns:
        df["Recall"] = pd.to_numeric(df["Recall"], errors="coerce")

    # Padronização da coluna de Lead Time
    if "Lead Time Avg" in df.columns and "lead_time_avg" not in df.columns:
        df["lead_time_avg"] = df["Lead Time Avg"]

    # Ordenação categórica personalizada
    existing_models = [m for m in df["model"].unique() if pd.notna(m)]
    ordered_cats = [m for m in MODEL_ORDER if m in existing_models] + [m for m in existing_models if m not in MODEL_ORDER]
    df["model"] = pd.Categorical(df["model"], categories=ordered_cats, ordered=True)

    return df


# ==========================================================
# 3. SUÍTE DE GRÁFICOS (APENAS DEEP LEARNING)
# ==========================================================
def plot_f1_score(df, output_dir=FIGURES_DIR):
    """Figura 1: F1-Score Direto da Rede Neural"""
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x="model", y="F1", hue="dataset", palette="Set1", edgecolor="black")
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8.5, xytext=(0, 3),
                        textcoords='offset points')

    plt.title("Desempenho de Classificação Direct/End-to-End: F1-Score por Arquitetura")
    plt.xlabel("Arquitetura Neural")
    plt.ylabel("F1-Score")
    plt.xticks(rotation=40, ha="right")
    plt.ylim(0, 1.1)
    plt.legend(title="Dataset", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_f1_score_dl.png"), dpi=300)
    plt.close()


def plot_auc_roc(df, output_dir=FIGURES_DIR):
    """Figura 2: AUC-ROC Direto da Rede Neural"""
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x="model", y="AUC", hue="dataset", palette="viridis", edgecolor="black")
    
    plt.title("Capacidade Discriminativa de Falha: Área Sob a Curva ROC (AUC-ROC)")
    plt.xlabel("Arquitetura Neural")
    plt.ylabel("AUC-ROC")
    plt.xticks(rotation=40, ha="right")
    plt.ylim(0.8, 1.02)
    plt.legend(title="Dataset", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_auc_roc_dl.png"), dpi=300)
    plt.close()


def plot_lead_time(df, output_dir=FIGURES_DIR):
    """Figura 3: Lead Time Médio de Detecção"""
    if "lead_time_avg" not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="model", y="lead_time_avg", hue="dataset", palette="Set2", edgecolor="black")

    plt.title("Antecedência Média de Detecção de Falha (Lead Time em Passos Temporais)")
    plt.xlabel("Arquitetura Neural")
    plt.ylabel("Lead Time (Passos)")
    plt.xticks(rotation=40, ha="right")
    plt.legend(title="Dataset", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_lead_time_dl.png"), dpi=300)
    plt.close()


def plot_forecast_metrics_if_available(df, output_dir=FIGURES_DIR):
    """Figura 4: Métricas de Regressão/Forecast (se disponíveis)"""
    forecast_cols = [c for c in ["Forecast_MSE", "Forecast_R2"] if c in df.columns and df[c].notna().any()]
    if not forecast_cols:
        return

    fig, axes = plt.subplots(1, len(forecast_cols), figsize=(7 * len(forecast_cols), 6))
    if len(forecast_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(forecast_cols):
        sns.barplot(data=df, x="model", y=col, hue="dataset", ax=axes[idx], edgecolor="black")
        axes[idx].set_title(f"Métrica de Regressão: {col}")
        axes[idx].set_xlabel("Modelo")
        axes[idx].tick_params(axis='x', rotation=40)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_forecast_metrics_dl.png"), dpi=300)
    plt.close()


# ==========================================================
# 4. EXECUÇÃO PRINCIPAL
# ==========================================================
def main():
    print("🚀 Processando resultados focando APENAS em Redes Neurais (Sem XGBoost)...")
    
    # Carregar e limpar
    df = load_and_clean_dl_data()

    # Salvar CSV filtrado
    cleaned_csv_path = os.path.join(SUMMARY_DIR, "dl_only_results.csv")
    df.to_csv(cleaned_csv_path, index=False)
    print(f"💾 Tabela purificada salva em: {cleaned_csv_path}")

    # Gerar gráficos
    print("🎨 Gerando gráficos focados para o relatório...")
    plot_f1_score(df)
    plot_auc_roc(df)
    plot_lead_time(df)
    plot_forecast_metrics_if_available(df)
    print("✅ Gráficos salvos com sucesso em results/summary/figures/")

    # Tabela formatada no terminal
    print("\n" + "="*85)
    print("🏆 RESULTADOS PURAGEM DAS REDES NEURAIS (ORDENADO POR F1-SCORE)")
    print("="*85)

    idx_best = df.groupby(["dataset", "model"], observed=True)["F1"].idxmax().dropna()
    best_df = df.loc[idx_best].sort_values(by=["dataset", "F1"], ascending=[True, False])

    cols_show = [c for c in ["dataset", "model", "F1", "AUC", "Precision", "Recall", "lead_time_avg"] if c in df.columns]
    print(best_df[cols_show].to_string(index=False))
    print("="*85 + "\n")


if __name__ == "__main__":
    main()