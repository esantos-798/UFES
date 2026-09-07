import numpy as np
import pandas as pd

# 2. Processo de Normalização (Min-Max Scaling)
# Criamos um novo DataFrame para armazenar os dados normalizados
df = pd.read_csv('dataset_treinamento.csv')

df_normalizado = pd.DataFrame()

for coluna in df.columns:
    col_min = df[coluna].min()
    col_max = df[coluna].max()
    
    # Aplicando a fórmula Min-Max
    df_normalizado[coluna] = (df[coluna] - col_min) / (col_max - col_min)

# 3. Salvando o dataset normalizado em um arquivo CSV
df_normalizado.to_csv('dataset_normalizado.csv', index=False)
print("Dataset normalizado salvo com sucesso!")
