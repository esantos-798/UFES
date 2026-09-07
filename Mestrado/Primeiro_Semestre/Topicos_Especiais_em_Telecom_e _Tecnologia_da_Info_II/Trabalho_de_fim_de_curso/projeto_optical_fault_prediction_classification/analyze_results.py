"""
==============================================================================
Analyze Results V2
Autor: Eduardo Ribeiro + ChatGPT
==============================================================================

Parte 1
- Descobre automaticamente todos os experiments
- Lê todos os metrics.json
- Cria summary.csv
- Cria consolidated_results.csv
==============================================================================

"""

import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ROOT = Path("results/runs")
SUMMARY = Path("results/summary")
FIGURES = SUMMARY / "figures"

SUMMARY.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# =============================================================================
# TODAS AS MÉTRICAS QUE PODEM EXISTIR
# =============================================================================

METRICS = [

    # Forecast
    "Forecast_MSE",
    "Forecast_RMSE",
    "Forecast_MAE",
    "Forecast_R2",
    "Forecast_Pearson",

    # Classification
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "AUC",

    "Threshold",

    "False Alarm Rate",
    "Miss Detection Rate",

    # Confusion Matrix
    "TN",
    "FP",
    "FN",
    "TP",

    # Lead Time
    "Average Lead Time",
    "Maximum Lead Time",
    "Minimum Lead Time",

    # Outros
    "Detected Failures",
    "Real Failures"
]

# =============================================================================
# AUXILIARES
# =============================================================================

def safe_get(dictionary, key):

    if key not in dictionary:
        return np.nan

    value = dictionary[key]

    if value is None:
        return np.nan

    return value


def parse_experiment_name(folder_name):
    """
    Extrai automaticamente:

    modelo
    dataset
    failure_weight
    pos_weight
    alpha
    """

    dataset = "unknown"

    if "_hard_failure_" in folder_name:
        dataset = "hard"

    if "_soft_failure_" in folder_name:
        dataset = "soft"

    model = folder_name

    if dataset == "hard":
        model = folder_name.split("_hard_failure")[0]

    if dataset == "soft":
        model = folder_name.split("_soft_failure")[0]

    fw = np.nan
    pw = np.nan
    alpha = np.nan

    m = re.search(r"_f(\d+)", folder_name)
    if m:
        fw = int(m.group(1))

    m = re.search(r"_p(\d+)", folder_name)
    if m:
        pw = int(m.group(1))

    m = re.search(r"_a([0-9.]+)", folder_name)
    if m:
        alpha = float(m.group(1))

    return {

        "model": model,
        "dataset": dataset,
        "failure_weight": fw,
        "pos_weight": pw,
        "alpha": alpha

    }


# =============================================================================
# LEITURA DOS EXPERIMENTOS
# =============================================================================

def scan_runs():

    rows = []

    if not ROOT.exists():

        print("Nenhuma pasta results/runs encontrada.")

        return pd.DataFrame()

    for run in ROOT.iterdir():

        if not run.is_dir():
            continue

        metrics_file = run / "metrics.json"

        if not metrics_file.exists():
            continue

        try:

            with open(metrics_file, "r") as f:

                metrics = json.load(f)

        except Exception as e:

            print(f"Erro lendo {metrics_file}: {e}")

            continue

        info = parse_experiment_name(run.name)

        row = {

            "experiment": run.name,

            **info

        }

        for metric in METRICS:

            row[metric] = safe_get(metrics, metric)

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


# =============================================================================
# CONSOLIDAÇÃO
# =============================================================================

def build_summary():

    print("=" * 70)
    print("LENDO EXPERIMENTOS")
    print("=" * 70)

    df = scan_runs()

    if df.empty:

        print("Nenhum experimento encontrado.")

        return df

    # ordenação

    df = df.sort_values(

        ["dataset",
         "model",
         "failure_weight"]

    )

    summary_file = SUMMARY / "summary.csv"

    consolidated_file = SUMMARY / "consolidated_results.csv"

    df.to_csv(summary_file, index=False)

    df.to_csv(consolidated_file, index=False)

    print()

    print(f"Resumo salvo em:\n{summary_file}")

    print(f"Consolidado salvo em:\n{consolidated_file}")

    print()

    print(df.head())

    return df

# =============================================================================
# MELHOR CONFIGURAÇÃO POR MODELO
# =============================================================================

def best_models(df):

    idx = df.groupby(
        ["dataset", "model"],
        observed=True
    )["F1"].idxmax()

    best = df.loc[idx].copy()

    best = best.sort_values(
        ["dataset", "F1"],
        ascending=[True, False]
    )

    return best


# =============================================================================
# TABELA PRINCIPAL DO ARTIGO
# =============================================================================

def generate_paper_results(df):

    best = best_models(df)

    cols = [

        "dataset",
        "model",
        "failure_weight",

        "F1",
        "AUC",

        "Forecast_MSE",
        "Forecast_RMSE",
        "Forecast_MAE",
        "Forecast_R2",
        "Forecast_Pearson",

        "Average Lead Time"

    ]

    cols = [c for c in cols if c in best.columns]

    paper = best[cols]

    output = SUMMARY / "paper_results.csv"

    paper.to_csv(output,index=False)

    print(f"Paper results salvo em {output}")

    return paper


# =============================================================================
# MELHOR CONFIGURAÇÃO GERAL
# =============================================================================

def generate_best_configuration(df):

    best = best_models(df)

    output = SUMMARY / "best_configuration.csv"

    best.to_csv(output,index=False)

    print(f"Best configuration salvo em {output}")


# =============================================================================
# COMPARAÇÃO ENTRE ARQUITETURAS
# =============================================================================

def generate_model_comparison(df):

    best = best_models(df)

    for dataset in ["hard","soft"]:

        sub = best[best.dataset==dataset]

        cols = [

            "model",
            "failure_weight",

            "F1",
            "AUC",

            "Forecast_RMSE",
            "Forecast_R2",

            "Average Lead Time"

        ]

        cols = [c for c in cols if c in sub.columns]

        sub = sub[cols]

        filename = SUMMARY / f"model_comparison_{dataset}.csv"

        sub.to_csv(filename,index=False)

        print(filename)


# =============================================================================
# SENSIBILIDADE AO FAILURE WEIGHT
# =============================================================================

def generate_failure_weight_tables(df):

    metrics = [

        "F1",

        "AUC",

        "Forecast_RMSE",

        "Forecast_R2",

        "Forecast_Pearson",

        "Average Lead Time"

    ]

    for dataset in df.dataset.unique():

        subset = df[df.dataset==dataset]

        for model in subset.model.unique():

            s = subset[subset.model==model]

            s = s.sort_values("failure_weight")

            cols = [

                "failure_weight"

            ] + [m for m in metrics if m in s.columns]

            table = s[cols]

            filename = SUMMARY / f"{dataset}_{model}_failure_weight.csv"

            table.to_csv(filename,index=False)


# =============================================================================
# EXPORTAÇÃO LATEX
# =============================================================================

def export_latex_tables(df):

    paper = generate_paper_results(df)

    latex = paper.to_latex(

        index=False,

        float_format="%.4f",

        caption="Resultados experimentais.",

        label="tab:results"

    )

    with open(

        SUMMARY/"paper_results.tex",

        "w",

        encoding="utf8"

    ) as f:

        f.write(latex)

    print("Tabela LaTeX criada.")

# =============================================================================
# MAIN
# =============================================================================

def main():

    print("="*70)
    print("ANALYZE RESULTS V2")
    print("="*70)

    # 1) Lê todas as runs
    df = build_summary()

    if df.empty:
        return

    print("\nGerando tabelas...")

    generate_paper_results(df)

    generate_best_configuration(df)

    generate_model_comparison(df)

    generate_failure_weight_tables(df)

    export_latex_tables(df)

    print("\nTudo concluído com sucesso!")



if __name__ == "__main__":
    main()    


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==========================================================
# CONFIG
# ==========================================================

INPUT = Path("results/summary/summary.csv")
OUT = Path("results/summary/paper_figures")

OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(INPUT)

print(df.columns)

# normalização nomes
df.columns = [c.strip() for c in df.columns]


# garantir modelo
if "model" not in df.columns:
    raise Exception("Coluna model não encontrada")


# ==========================================================
# FIGURA 1
# F1 SCORE COMPARAÇÃO FINAL
# ==========================================================

plt.figure(figsize=(12,6))

best = (
    df.groupby(["dataset","model"])
      ["F1"]
      .max()
      .reset_index()
)


sns.barplot(
    data=best,
    x="F1",
    y="model",
    hue="dataset"
)

plt.title(
    "Performance Comparison of Architectures\nBest F1-score"
)

plt.xlabel("F1-score")
plt.ylabel("Architecture")

plt.xlim(0,1)

plt.tight_layout()

plt.savefig(
    OUT/"fig1_f1_comparison.png",
    dpi=300
)

plt.close()



# ==========================================================
# FIGURA 2
# SENSIBILIDADE AO FAILURE WEIGHT
# ==========================================================

if "failure_weight" in df.columns:

    multitask = df[
        df["model"].str.contains(
            "multitask",
            case=False,
            na=False
        )
    ]


    plt.figure(figsize=(10,6))


    sns.lineplot(
        data=multitask,
        x="failure_weight",
        y="F1",
        hue="model",
        style="dataset",
        marker="o"
    )


    plt.title(
        "Sensitivity Analysis of Failure Weight β"
    )

    plt.xlabel("Failure Weight β")
    plt.ylabel("F1-score")


    plt.tight_layout()

    plt.savefig(
        OUT/"fig2_beta_sensitivity.png",
        dpi=300
    )

    plt.close()



# ==========================================================
# FIGURA 3
# PEARSON FORECAST
# ==========================================================


pearson = [
    c for c in df.columns
    if "pearson" in c.lower()
]


if pearson:

    pcol = pearson[0]


    best_p = (
        df.groupby(["dataset","model"])
        [pcol]
        .max()
        .reset_index()
    )


    plt.figure(figsize=(12,6))


    sns.barplot(
        data=best_p,
        x=pcol,
        y="model",
        hue="dataset"
    )


    plt.title(
        "Forecast Quality - Pearson Correlation"
    )


    plt.xlabel("Pearson Correlation")
    plt.ylabel("Architecture")


    plt.xlim(0,1)

    plt.tight_layout()


    plt.savefig(
        OUT/"fig3_pearson.png",
        dpi=300
    )

    plt.close()



# ==========================================================
# FIGURA 4
# PEARSON vs F1
# ==========================================================


if pearson:

    pcol=pearson[0]


    plt.figure(figsize=(8,6))


    sns.scatterplot(
        data=df,
        x=pcol,
        y="F1",
        hue="model",
        style="dataset",
        s=150
    )


    plt.title(
        "Relationship Between Forecast and Detection Performance"
    )

    plt.xlabel(
        "Pearson Forecast Correlation"
    )

    plt.ylabel(
        "F1-score"
    )


    plt.xlim(0,1)
    plt.ylim(0,1)


    plt.tight_layout()


    plt.savefig(
        OUT/"fig4_pearson_vs_f1.png",
        dpi=300
    )

    plt.close()



# ==========================================================
# FIGURA 5
# LEAD TIME
# ==========================================================


lead_cols=[
    c for c in df.columns
    if "lead" in c.lower()
]


if lead_cols:

    lcol=lead_cols[0]


    best_l = (
        df.groupby(["dataset","model"])
        [lcol]
        .max()
        .reset_index()
    )


    plt.figure(figsize=(12,6))


    sns.barplot(
        data=best_l,
        x=lcol,
        y="model",
        hue="dataset"
    )


    plt.title(
        "Average Failure Detection Lead Time"
    )


    plt.xlabel(
        "Lead Time"
    )

    plt.ylabel(
        "Architecture"
    )


    plt.tight_layout()


    plt.savefig(
        OUT/"fig5_lead_time.png",
        dpi=300
    )

    plt.close()



print("\n================================")
print("FIGURAS GERADAS COM SUCESSO")
print("Local:")
print(OUT)
print("================================")    