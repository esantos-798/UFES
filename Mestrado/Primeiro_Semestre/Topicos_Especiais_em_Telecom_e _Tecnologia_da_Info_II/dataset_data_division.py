import pandas as pd

# 1. Embaralhar os dados de forma aleatória para evitar viés de ordenação
# (A semente 'random_state=42' garante que o embaralhamento seja reprodutível)
df_norm = pd.read_csv('dataset_normalizado.csv')

df_shuffled = df_norm.sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Definindo os pontos de corte para 70% / 15% / 15%
train_end = int(0.70 * len(df_shuffled))         # Índice 700
val_end = train_end + int(0.15 * len(df_shuffled)) # Índice 850

# 3. Dividindo o DataFrame fatiando as linhas (Slicing)
df_train = df_shuffled.iloc[:train_end]            # Linhas 0 a 699 (700 amostras)
df_val = df_shuffled.iloc[train_end:val_end]       # Linhas 700 a 849 (150 amostras)
df_test = df_shuffled.iloc[val_end:]               # Linhas 850 a 999 (150 amostras)

# 4. Salvando os três arquivos CSV independentes
df_train.to_csv('dataset_treinamento_split.csv', index=False)
df_val.to_csv('dataset_validacao_split.csv', index=False)
df_test.to_csv('dataset_teste_split.csv', index=False)