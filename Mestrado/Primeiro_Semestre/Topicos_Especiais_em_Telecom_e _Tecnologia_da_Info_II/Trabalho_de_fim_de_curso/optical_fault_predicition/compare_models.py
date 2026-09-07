import json
from pathlib import Path

import pandas as pd

from src.utils.visualization import plot_model_comparison

EXPERIMENTS = Path("experiments")

results = []

for folder in EXPERIMENTS.iterdir():

    metrics_file = folder / "metrics.json"

    if metrics_file.exists():

        with open(metrics_file) as f:
            metrics = json.load(f)

        # adiciona automaticamente o nome do modelo
        metrics["Model"] = folder.name.upper()

        results.append(metrics)

df = pd.DataFrame(results)

# Coloca a coluna Model na frente
cols = ["Model"] + [c for c in df.columns if c != "Model"]
df = df[cols]

df = df.sort_values("F1", ascending=False)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================\n")

print(df.to_string(index=False))

print(df)
print(df.columns)

plot_model_comparison(df)