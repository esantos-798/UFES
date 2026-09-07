# generate_soft_failure_data.py
import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)
np.random.seed(42)

timestamps_count = 1200 
num_devices = 15

time_range = pd.date_range(start="2026-01-01", periods=timestamps_count, freq="5min")
rows = []

print("Gerando DESAFIO CRÍTICO: Rampa ultralenta com ruído severo...")

for t_idx, t in enumerate(time_range):
    # Rampa MUITO mais lenta e sutil (máximo de 0.11 de queda em vez de 0.35)
    if t_idx >= (timestamps_count // 2):
        factor = 0.11 * (t_idx - (timestamps_count // 2)) / (timestamps_count // 2)
    else:
        factor = 0.0
        
    for dev_id in range(num_devices):
        # AUMENTO DO RUÍDO: Escala subiu de 0.02 para 0.065 (Ruído severo na linha)
        base_power = 1.0 + np.random.normal(0, 0.065)
        base_osnr = 22.0 + np.random.normal(0, 0.2)
        base_ber = 1e-9 + np.random.uniform(1e-10, 5e-10)
        
        if dev_id == 0:
            base_power -= factor
            base_osnr -= factor * 2.0
            base_ber += factor * 1e-5
        elif dev_id == 1:
            base_power -= factor * 0.85
            base_osnr -= factor * 1.5
        elif dev_id == 2:
            base_power -= factor * 0.50
            
        rows.append({
            'Timestamp': t,
            'ID': f"Ampli{dev_id}",
            'InputPower': base_power + 0.2,
            'OutputPower': base_power,
            'OSNR': base_osnr,
            'BER': base_ber
        })

df_raw = pd.DataFrame(rows)
df_pivot = df_raw.pivot_table(index='Timestamp', columns='ID', values=['BER', 'OSNR', 'InputPower', 'OutputPower'])
df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
df_pivot = df_pivot.reset_index()

df_pivot.to_csv("data/SoftFailure_dataset.csv", index=False)
print("[SUCESSO] Novo dataset complexo de Soft Failure gerado!")