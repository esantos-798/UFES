import numpy as np
import pandas as pd

# 1. Definindo a função com 4 variáveis de entrada independentes
def calcular_saida(x1, x2, x3, x4):
    return 2.5 * x1 + 1.2 * (x2 ** 2) - 3.0 * np.sin(x3) + 0.8 * x4

# Configurando a semente aleatória para reprodutibilidade
np.random.seed(42)

# 2. Gerando 1000 padrões de treinamento (dados fictícios bem distribuídos)
n_samples = 1000
x1_vals = np.random.uniform(0, 10, n_samples)      # Distribuição uniforme de 0 a 10
x2_vals = np.random.uniform(1, 5, n_samples)       # Distribuição uniforme de 1 a 5
x3_vals = np.random.uniform(0, np.pi, n_samples)   # Distribuição de 0 a Pi (para o seno)
x4_vals = np.random.normal(50, 15, n_samples)      # Distribuição normal (média 50, desvio 15)

# Calculando a saída correspondente através da função
y_vals = calcular_saida(x1_vals, x2_vals, x3_vals, x4_vals)

# Adicionando ruído gaussiano para tornar o desafio de modelagem mais realista
ruido = np.random.normal(0, 1, n_samples)
y_vals_com_ruido = y_vals + ruido

# 3. Montando o Dataset estruturado
df = pd.DataFrame({
    'Var_A': x1_vals,
    'Var_B': x2_vals,
    'Var_C': x3_vals,
    'Var_D': x4_vals,
    'Alvo_Y': y_vals_com_ruido
})

# 4. Armazenando em um arquivo CSV
df.to_csv('dataset_treinamento.csv', index=False)