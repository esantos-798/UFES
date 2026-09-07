import glob
import json
import os
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Caminhos e Mapeamento de Modelos
# -----------------------------------------------------------------------------
RUNS_DIR = r"results\runs"
OUTPUT_CSV = r"results\summary\consolidated_results_cleaned.csv"

model_mapping = {
    "MultiTask + Transformer": "multitask_lstnet_transformer",
    "MultiTask + TCN": "multitask_lstnet_tcn",
    "MultiTask-LSTNet": "multitask_lstnet",
    "MultiTask + Attention": "multitask_lstnet_attention",
    "TCN": "tcn",
    "GRU": "gru",
    "Transformer": "transformer",
    "BiLSTM": "bilstm",
    "LSTM": "lstm",
    "LSTNet": "lstnet",
    "Attention-LSTNet": "attention_lstnet",
}

# -----------------------------------------------------------------------------
# 2. Leitura Exclusiva dos arquivos metrics.json
# -----------------------------------------------------------------------------
print("🔍 Buscando exclusivamente por arquivos metrics.json em results/runs...")

metrics_files = glob.glob(
    os.path.join(RUNS_DIR, "**", "metrics.json"), recursive=True
)

rows = []
for file_path in metrics_files:
  try:
    with open(file_path, "r", encoding="utf-8") as f:
      data = json.load(f)

      if isinstance(data, dict):
        data["_file_path"] = file_path
        rows.append(data)
      elif isinstance(data, list):
        for item in data:
          if isinstance(item, dict):
            item["_file_path"] = file_path
            rows.append(item)
  except Exception:
    continue

if not rows:
  raise FileNotFoundError(
      "Nenhum arquivo 'metrics.json' foi encontrado em results/runs!"
  )

consolidated_df = pd.DataFrame(rows)

# Normalizar colunas para minúsculas
orig_cols = list(consolidated_df.columns)
col_rename = {c: c.strip().lower() for c in orig_cols}
consolidated_df = consolidated_df.rename(columns=col_rename)

# Ajuste da coluna Dataset
if "dataset" not in consolidated_df.columns:
  consolidated_df["dataset"] = consolidated_df["_file_path"].apply(
      lambda p: (
          "hard" if "hard" in p.lower() else ("soft" if "soft" in p.lower() else "unknown")
      )
  )

# Salva o arquivo limpo e consolidado
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
consolidated_df.to_csv(OUTPUT_CSV, index=False)
print(
    f"✅ {len(consolidated_df)} execuções extraídas dos 'metrics.json' e salvas"
    f" em: {OUTPUT_CSV}\n"
)


# -----------------------------------------------------------------------------
# 3. Extração das Métricas Pearson e R2 (Tratando Escalares e Arrays)
# -----------------------------------------------------------------------------
def get_val(df_subset, keys):
  """Busca o primeiro valor válido e retorna como escalar de forma segura."""
  for k in keys:
    k_lower = k.lower()
    if k_lower in df_subset.columns:
      vals = df_subset[k_lower].dropna()
      if not vals.empty:
        # Pega o primeiro valor do array/série
        val = vals.iloc[0]
        try:
          val_float = float(val)
          if not pd.isna(val_float):
            return val_float
        except (ValueError, TypeError):
          continue
  return None


def format_metric(val):
  if val is None:
    return "N/A"
  return f"{val:.3f}"


def build_summary_table(ref_list, dataset_type):
  table_data = []

  # Filtra pelo dataset (hard ou soft)
  df_ds = consolidated_df[consolidated_df["dataset"] == dataset_type]

  for model_display, weight, f1, auc in ref_list:
    model_key = model_mapping[model_display]

    # Busca pelo nome do modelo na coluna 'model' ou no caminho do arquivo
    match = (
        df_ds[df_ds["model"] == model_key]
        if "model" in df_ds.columns
        else pd.DataFrame()
    )

    if match.empty:
      match = df_ds[df_ds["_file_path"].str.contains(model_key, na=False)]

    # Filtra pelo peso de falha (failure_weight)
    if not match.empty and "failure_weight" in match.columns:
      match_w = match[match["failure_weight"] == weight]
      if not match_w.empty:
        match = match_w

    # Busca do Pearson e R2
    pearson = get_val(
        match,
        [
            "forecast_pearson",
            "pearson",
            "forecast_pearson_avg",
            "pearson_correlation",
        ],
    )
    r2 = get_val(match, ["forecast_r2", "r2", "forecast_r2_avg"])

    table_data.append({
        "Modelo": model_display,
        "Weight": weight,
        "F1": f1,
        "AUC": auc,
        "Pearson": format_metric(pearson),
        "R2": format_metric(r2),
    })

  return pd.DataFrame(table_data)


# -----------------------------------------------------------------------------
# 4. Dados de Referência e Impressão
# -----------------------------------------------------------------------------
hard_ref = [
    ("MultiTask + Transformer", 70, 0.452, 0.763),
    ("MultiTask + TCN", 20, 0.451, 0.764),
    ("MultiTask-LSTNet", 20, 0.434, 0.794),
    ("MultiTask + Attention", 20, 0.420, 0.765),
    ("TCN", 20, 0.415, 0.765),
    ("GRU", 20, 0.410, 0.762),
    ("Transformer", 50, 0.409, 0.785),
    ("BiLSTM", 150, 0.406, 0.773),
    ("LSTM", 150, 0.404, 0.790),
    ("LSTNet", 70, 0.403, 0.774),
    ("Attention-LSTNet", 100, 0.389, 0.795),
]

soft_ref = [
    ("MultiTask + Transformer", 30, 0.956, 0.986),
    ("MultiTask + TCN", 70, 0.956, 0.985),
    ("MultiTask-LSTNet", 150, 0.956, 0.988),
    ("MultiTask + Attention", 30, 0.926, 0.988),
    ("BiLSTM", 70, 0.404, 0.713),
    ("LSTM", 100, 0.393, 0.711),
    ("LSTNet", 10, 0.378, 0.674),
    ("GRU", 100, 0.373, 0.714),
    ("Attention-LSTNet", 20, 0.341, 0.710),
    ("TCN", 50, 0.329, 0.774),
    ("Transformer", 50, 0.324, 0.811),
]

df_hard = build_summary_table(hard_ref, "hard")
df_soft = build_summary_table(soft_ref, "soft")

print("--- HARD FAILURE ---")
print(df_hard.to_string(index=False))

print("\n--- SOFT FAILURE ---")
print(df_soft.to_string(index=False))