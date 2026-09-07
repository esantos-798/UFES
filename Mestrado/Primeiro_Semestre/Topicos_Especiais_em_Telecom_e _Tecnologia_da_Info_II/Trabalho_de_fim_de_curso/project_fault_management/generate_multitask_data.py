# generate_multitask_data.py
import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
np.random.seed(42)

timestamps_count = 3000
num_devices = 4  # Vamos focar nos 4 amplificadores críticos para o teste
time_range = pd.date_range(start="2026-01-01", periods=timestamps_count, freq="5min")

rows = []
print("Gerando Dataset Unificado Multitarefa (Normal, Hard, Soft)...")

for t_idx, t in enumerate(time_range):
    # Base saudável ruidosa
    base_powers = {dev: 1.0 + np.random.normal(0, 0.03) for dev in range(num_devices)}
    
    # Labels padrão (0: Não/Nenhum/Normal)
    label_det = 0
    label_loc = 0  # 0 = Nenhum, 1 = Ampli0, 2 = Ampli1, etc.
    label_sev = 0  # 0 = Normal, 1 = Soft, 2 = Hard
    
    # Bloco 1: De 1000 a 1500 -> Injetar Soft Failure no Ampli1 (ID físico Ampli1, label_loc = 2)
    if 1000 <= t_idx < 1500:
        factor = 0.25 * (t_idx - 1000) / 500
        base_powers[1] -= factor      # Causa raiz no Ampli1
        base_powers[2] -= factor * 0.7 # Propagação para os seguintes
        base_powers[3] -= factor * 0.5
        label_det = 1
        label_loc = 2  # Falha no Ampli1
        label_sev = 1  # Soft
        
    # Bloco 2: De 2000 a 2500 -> Injetar Hard Failure abrupto no Ampli3 (label_loc = 4)
    elif 2000 <= t_idx < 2500:
        base_powers[3] -= 0.6  # Queda abrupta instantânea em degrau
        label_det = 1
        label_loc = 4  # Falha no Ampli3
        label_sev = 2  # Hard
        
    for dev_id in range(num_devices):
        rows.append({
            'Timestamp': t,
            'ID': f"Ampli{dev_id}",
            'InputPower': base_powers[dev_id] + 0.2,
            'OutputPower': base_powers[dev_id],
            'OSNR': 22.0 + np.random.normal(0, 0.1),
            'BER': 1e-9,
            'Label_Det': label_det,
            'Label_Loc': label_loc,
            'Label_Sev': label_sev
        })

df_raw = pd.DataFrame(rows)

# Mapeia as labels associando-as ao timestamp (uma linha por timestep após o pivot)
df_labels = df_raw.groupby('Timestamp')[['Label_Det', 'Label_Loc', 'Label_Sev']].first().reset_index()

df_pivot = df_raw.pivot_table(index='Timestamp', columns='ID', values=['InputPower', 'OutputPower', 'OSNR', 'BER'])
df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
df_pivot = df_pivot.reset_index()

# Junta variáveis e labels em um único arquivo estruturado
df_final = pd.merge(df_pivot, df_labels, on='Timestamp')
df_final.to_csv("data/Multitask_dataset.csv", index=False)
print("[SUCESSO] Dataset 'data/Multitask_dataset.csv' gerado para o Monstro Multitarefa!")