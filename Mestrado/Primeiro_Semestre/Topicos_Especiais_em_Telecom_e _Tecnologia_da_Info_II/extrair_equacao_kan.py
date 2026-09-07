import pandas as pd
import numpy as np
import torch
from kan import KAN

print("=== Iniciando a Regressão Simbólica via KAN (Corrigido) ===")

# 1. Carregar os datasets
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = torch.tensor(df_train[colunas_x].values, dtype=torch.float32)
y_train = torch.tensor(df_train['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(df_test[colunas_x].values, dtype=torch.float32)
y_test = torch.tensor(df_test['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)

dataset = {'train_input': X_train, 'train_label': y_train, 'test_input': X_test, 'test_label': y_test}

# 2. Instanciar e ajustar o modelo
modelo_kan = KAN(width=[4, 2, 1], grid=5, k=3, seed=42)

if hasattr(modelo_kan, 'fit'):
    modelo_kan.fit(dataset, steps=20)
else:
    modelo_kan.train(dataset, steps=20)

# 3. Passos da Regressão Simbólica
print("\n-> Executando Pruning (Poda de conexões)...")
modelo_kan.prune()

print("-> Identificando as funções matemáticas candidatas...")
modelo_kan.auto_symbolic()

print("\n=========================================================================")
print("FORMULAÇÃO MATEMÁTICA ENCONTRADA PELA KAN:")
print("=========================================================================")

# Chamada corrigida compatível com MultKAN (retorna as fórmulas simbólicas em formato SymPy)
try:
    formulas = modelo_kan.symbolic_formula()
    # Exibe a fórmula para a nossa única saída (índice 0)
    print("Alvo_Y = ", formulas[0][0])
except Exception:
    # Fallback caso a estrutura mude ligeiramente
    print(modelo_kan.suggest_symbolic())

print("=========================================================================")