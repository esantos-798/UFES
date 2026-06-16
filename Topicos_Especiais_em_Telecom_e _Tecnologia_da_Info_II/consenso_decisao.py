import pandas as pd
import numpy as np

print("Iniciando o Algoritmo de Decisão por Consenso...\n")

# 1. Definir os dados extraídos das 3 análises anteriores
variaveis = ['Var_A', 'Var_B', 'Var_C', 'Var_D']

# Critério 1: Erro por Permutação (Impacto real medido no gráfico de barras)
# Valores aproximados do seu MSE obtido no gráfico
impacto_mse = np.array([0.0031, 0.0005, 0.0007, 0.0016])

# Critério 2: Coeficientes Absolutos da Equação KAN (Magnitude matemática)
# Pegamos o peso do termo linear principal de cada uma na fórmula
pesos_kan = np.array([0.2216, 0.2452, 0.00007, 0.7630])

# Critério 3: Acoplamento de Variância (1 / Desvio Padrão)
# Como o desvio menor indicou maior dominância (Var_D), usamos o inverso do desvio
desvios = np.array([0.2939, 0.2943, 0.2944, 0.1456])
forca_variancia = 1.0 / desvios

# 2. Normalizar os critérios (Escala de 0 a 1) para que todos tenham o mesmo peso no consenso
def normalizar(array):
    return (array - array.min()) / (array.max() - array.min())

norm_mse = normalizar(impacto_mse)
norm_kan = normalizar(pesos_kan)
norm_var = normalizar(forca_variancia)

# 3. Calcular o Score de Consenso (Média ponderada ou simples dos 3 fatores)
# Daremos pesos iguais (1/3) para a equação, o erro prático e a estatística pura
score_consenso = (norm_mse + norm_kan + norm_var) / 3.0

# 4. Montar o DataFrame do Veredito
df_consenso = pd.DataFrame({
    'Variável': variaveis,
    'Score Permutação (Norm)': norm_mse,
    'Score Fórmula KAN (Norm)': norm_kan,
    'Score Variância (Norm)': norm_var,
    'Score Consenso Final': score_consenso
})

# Ordenar o ranking do consenso
df_consenso = df_consenso.sort_values(by='Score Consenso Final', ascending=False).reset_index(drop=True)

print("========================================================")
print("RANKING MATRIZ DE CONSENSO FINAL:")
print(df_consenso.round(4).to_string())
print("========================================================")

# 5. Salvar o veredito para o relatório
df_consenso.round(4).to_csv('matriz_consenso_ufes.csv', sep=';', index=False)
print("\nMatriz de consenso salva em 'matriz_consenso_ufes.csv'!")