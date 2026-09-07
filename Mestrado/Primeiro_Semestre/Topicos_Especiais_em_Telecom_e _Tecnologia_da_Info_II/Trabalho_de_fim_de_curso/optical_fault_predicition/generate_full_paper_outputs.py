import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de diretórios
RESULTS_DIR = "results"
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
FIGURES_DIR = os.path.join(SUMMARY_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'figure.titlesize': 14})

def load_data():
    """Carrega grid_results.csv ou consolidated_results_cleaned.csv removendo colunas duplicadas"""
    if os.path.exists("grid_results.csv"):
        df = pd.read_csv("grid_results.csv")
    elif os.path.exists("consolidated_results_cleaned.csv"):
        df = pd.read_csv("consolidated_results_cleaned.csv")
    else:
        found = glob.glob("**/grid_results.csv", recursive=True) + glob.glob("**/summary.csv", recursive=True)
        if not found: 
            raise FileNotFoundError("Nenhum arquivo de resultados (grid_results.csv / summary.csv) encontrado!")
        df = pd.read_csv(found[0])

    # 1. Limpeza de espaços nos nomes das colunas
    df.columns = [c.strip() for c in df.columns]
    
    # 2. REMOVE COLUNAS DUPLICADAS (Mantém apenas a primeira ocorrência)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Tratamento de datasets
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.replace("_failure", "")
    
    # Padronização de colunas numéricas
    cols_map = {
        'forecast_r2': 'Forecast_R2',
        'forecast_pearson': 'Forecast_Pearson',
        'forecast_mse': 'Forecast_MSE',
        'lead_time_avg': 'Lead_Time_Avg',
        'Average Lead Time': 'Lead_Time_Avg'
    }
    df = df.rename(columns=cols_map)

    # Remove duplicadas novamente caso o rename tenha gerado colisões
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Conversão segura para numérico
    for col in ['F1', 'AUC', 'Forecast_R2', 'Forecast_Pearson', 'Forecast_MSE', 'Lead_Time_Avg', 'MDR', 'FAR', 'failure_weight']:
        if col in df.columns:
            # Força o tratamento como Series caso ainda haja ambiguidade
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            df[col] = pd.to_numeric(series, errors='coerce')

    return df

def generate_summary_and_tables(df):
    """Gera summary.csv, paper_full_results.csv e paper_full_table.tex"""
    # Agrupa e extrai o melhor F1 por modelo e dataset
    idx_best = df.groupby(["dataset", "model"], observed=True)["F1"].idxmax().dropna()
    summary_df = df.loc[idx_best].copy()

    # Salva CSVs
    summary_csv = os.path.join(SUMMARY_DIR, "summary.csv")
    paper_csv = os.path.join(SUMMARY_DIR, "paper_full_results.csv")
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_csv(paper_csv, index=False)
    print(f"💾 Salvos: {summary_csv} e {paper_csv}")

    # Exporta tabela LaTeX
    tex_path = os.path.join(SUMMARY_DIR, "paper_full_table.tex")
    show_cols = [c for c in ['dataset', 'model', 'F1', 'AUC', 'Forecast_R2', 'Forecast_Pearson', 'Lead_Time_Avg', 'MDR', 'FAR'] if c in summary_df.columns]
    
    latex_str = summary_df[show_cols].to_latex(
        index=False,
        float_format="%.4f",
        caption="Resumo de Desempenho do Modelo nas Tarefas Multitarefa e Detecção de Falhas",
        label="tab:full_results"
    )
    
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"📄 Tabela LaTeX exportada em: {tex_path}")

    return summary_df

def plot_pareto(summary_df):
    """Gera o Gráfico de Pareto: F1-Score vs Lead Time"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=summary_df, x="Lead_Time_Avg", y="F1", 
        hue="model", style="dataset", s=140, palette="tab10"
    )
    plt.title("Fronteira de Pareto: Qualidade de Detecção (F1) vs Antecedência (Lead Time)")
    plt.xlabel("Antecedência Média de Aviso (Lead Time em Passos)")
    plt.ylabel("F1-Score (Classificação)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_pareto_f1_vs_leadtime.png"), dpi=300)
    plt.close()

def plot_sensitivity(df):
    """Gera a Análise de Sensibilidade do Peso da Perda (Failure Weight)"""
    if "failure_weight" not in df.columns or df["failure_weight"].nunique() <= 1:
        return
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df, x="failure_weight", y="F1", 
        hue="dataset", marker="o", linewidth=2.5
    )
    plt.title("Análise de Sensibilidade: Impacto do Failure Weight no F1-Score")
    plt.xlabel("Peso da Perda de Falha (Failure Weight)")
    plt.ylabel("F1-Score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_failure_weight_sensitivity.png"), dpi=300)
    plt.close()

def plot_pearson_r2_forecast(summary_df):
    """Gera a dispersão real de R2 vs Pearson para comparação com o artigo"""
    if "Forecast_Pearson" in summary_df.columns and "Forecast_R2" in summary_df.columns:
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=summary_df, x="Forecast_Pearson", y="Forecast_R2", 
            hue="model", style="dataset", s=130, palette="viridis"
        )
        plt.title("Qualidade de Predição Contínua (Forecast): Correlação de Pearson vs R²")
        plt.xlabel("Correlação de Pearson (Forecast)")
        plt.ylabel("Coeficiente de Determinação (R²)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "07_pearson_vs_r2.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    print("🚀 Gerando suite completa de relatórios, gráficos e tabelas para o artigo...")
    raw_df = load_data()
    best_df = generate_summary_and_tables(raw_df)
    
    plot_pareto(best_df)
    plot_sensitivity(raw_df)
    plot_pearson_r2_forecast(best_df)
    
    print("✅ Todos os artefatos (Pareto, Sensitivity, Pearson/R2, summary.csv e .tex) foram reconstruídos com sucesso!")