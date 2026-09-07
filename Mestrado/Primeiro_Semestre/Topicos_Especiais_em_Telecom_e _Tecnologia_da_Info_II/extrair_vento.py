import pandas as pd
import numpy as np

# 1. Carrega o dataset
df = pd.read_csv("INMET_SE_ES_A612_VITORIA_01-01-2025_A_31-12-2025.CSV", sep=';', skiprows=8, encoding='latin-1', decimal=',')

# --- CORREÇÃO DA HORA ---
# Vamos listar as colunas para você ver no terminal exatamente o que veio no seu arquivo
print("Colunas encontradas no arquivo:", df.columns.tolist())

# O INMET costuma usar 'Hora (UTC)' ou 'HORA (UTC)'. Vamos tentar identificar dinamicamente:
coluna_data = [c for c in df.columns if 'Data' in c or 'DATA' in c][0]
coluna_hora = [c for c in df.columns if 'Hora' in c or 'HORA' in c][0]

print(f"\nUsando a coluna de data: '{coluna_data}' e de hora: '{coluna_hora}'")

# 2. Junta as colunas usando os nomes detectados automaticamente
# (Ajustei o format para ignorar o 'UTC' fixo no final, deixando mais flexível)
df['Data_Hora'] = pd.to_datetime(df[coluna_data] + ' ' + df[coluna_hora], errors='coerce')
df.set_index('Data_Hora', inplace=True)

# 3. Substitui os erros/valores ausentes (9999) por NaN
df = df.replace([-9999, 9999, -9999.0, 9999.0], np.nan)

# --- CORREÇÃO DO VENTO ---
# Identifica a coluna de vento dinamicamente para evitar outro KeyError
coluna_vento = [c for c in df.columns if 'VENTO' in c.upper() and 'VELOCIDADE' in c.upper()][0]
print(f"Usando a coluna de vento: '{coluna_vento}'")

# 4. Extrai apenas o vento
df_vento = df[[coluna_vento]].copy()

print("\nPrimeiras linhas do dataset de vento:")
print(df_vento.head())