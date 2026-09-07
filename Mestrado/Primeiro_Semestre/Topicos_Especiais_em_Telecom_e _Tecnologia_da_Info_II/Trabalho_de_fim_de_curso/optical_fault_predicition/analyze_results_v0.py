import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results/runs")

rows = []

for run in ROOT.iterdir():

    metrics_file = run / "metrics.json"

    if not metrics_file.exists():
        continue

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    cfg = run.name

    dataset = "hard" if "hard_failure" in cfg else "soft"

    fw = int(cfg.split("_f")[-1].split("_")[0])

    rows.append({
        "experiment": cfg,
        "dataset": dataset,
        "failure_weight": fw,
        **metrics
    })

df = pd.DataFrame(rows)

df = df.sort_values(["dataset","failure_weight"])

output = Path("results/summary")
output.mkdir(parents=True, exist_ok=True)

df.to_csv(output/"summary.csv", index=False)

print(df)

print("\nSaved:")
print(output/"summary.csv")

metrics = [
    "F1",
    "AUC",
    "Precision",
    "Recall",
    "Forecast_R2",
    "Forecast_RMSE",
    "Average Lead Time"
]

for dataset in df.dataset.unique():

    subset = df[df.dataset == dataset]

    for metric in metrics:

        if metric not in subset.columns:
            continue

        plt.figure(figsize=(6,4))

        plt.plot(
            subset["failure_weight"],
            subset[metric],
            marker="o"
        )

        plt.title(f"{dataset} - {metric}")
        plt.xlabel("Failure Weight")
        plt.ylabel(metric)
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(output/f"{dataset}_{metric}.png")

        plt.close()

print("\nGraphs generated.")

print("\n==============================")
print("BEST EXPERIMENTS")
print("==============================")

for dataset in df.dataset.unique():

    subset = df[df.dataset==dataset]

    best_f1 = subset.loc[subset["F1"].idxmax()]

    print()

    print(dataset.upper())

    print(best_f1["experiment"])

    print(f"F1        : {best_f1['F1']:.4f}")
    print(f"AUC       : {best_f1['AUC']:.4f}")
    print(f"Precision : {best_f1['Precision']:.4f}")
    print(f"Recall    : {best_f1['Recall']:.4f}")