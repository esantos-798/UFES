import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from kan import MultKAN

print("Iniciando o Super Algoritmo de Decisão por Consenso...\n")

# 1. Carregar os dados de teste
df_test = pd.read_csv('dataset_teste_split.csv')
variaveis = ['Var_A', 'Var_B', 'Var_C', 'Var_D']

X_test_np = df_test[variaveis].values
y_test_np = df_test['Alvo_Y'].values
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

# ==========================================
# MÉTODO 1: COEFICIENTE DE PEARSON (Linear)
# ==========================================
print("-> Calculando Correlação de Pearson...")
pearsons = [abs(df_test[var].corr(df_test['Alvo_Y'])) for var in variaveis]

# ==========================================
# MÉTODO 2: MUTUAL INFORMATION (Não-linear)
# ==========================================
print("-> Calculando Informação Mútua...")
mi_scores = mutual_info_regression(X_test_np, y_test_np, random_state=42)

# ==========================================
# MÉTODO 3: PERMUTATION IMPORTANCE (Estresse)
# ==========================================
print("-> Calculando Permutation Importance via KAN...")
# Reconstruir e carregar a KAN
model = MultKAN(width=[4, 10, 1], grid=5, k=3)
try:
    # Ajustado de load_ckpt para loadckpt
    model.loadckpt('0.1')
except:
    try:
        model.loadckpt('0.2')
    except Exception as e:
        print(f"Nota: Não foi possível carregar os pesos: {e}")

with torch.no_grad():
    y_pred_base = model(X_test_tensor).numpy()
erro_base = mean_squared_error(y_test_np, y_pred_base)

importancias_mse = []
for idx, var in enumerate(variaveis):
    X_test_permutado = X_test_np.copy()
    np.random.shuffle(X_test_permutado[:, idx])
    X_perm_tensor = torch.tensor(X_test_permutado, dtype=torch.float32)
    with torch.no_grad():
        y_pred_perm = model(X_perm_tensor).numpy()
    importancias_mse.append(mean_squared_error(y_test_np, y_pred_perm) - erro_base)

# ==========================================
# MÉTODO 4: MÁXIMO COEFICIENTE DA FÓRMULA KAN
# ==========================================
# Pesos extraídos diretamente da equação final da sua rede
pesos_kan = np.array([0.2216, 0.2452, 0.00007, 0.7630])

# ==========================================
# MÉTODO 5: ACOPLAMENTO DE VARIÂNCIA (1/std)
# ==========================================
desvios = np.array([0.2939, 0.2943, 0.2944, 0.1456])
forca_variancia = 1.0 / desvios

# ==========================================
# CENTRALIZAÇÃO E NORMALIZAÇÃO DA MATRIZ
# ==========================================
def normalizar(array):
    if (array.max() - array.min()) == 0:
        return np.zeros_like(array)
    return (array - array.min()) / (array.max() - array.min())

norm_pearson = normalizar(np.array(pearsons))
norm_mi = normalizar(np.array(mi_scores))
norm_mse = normalizar(np.array(importancias_mse))
norm_kan = normalizar(pesos_kan)
norm_var = normalizar(forca_variancia)

# Adicione a nova métrica extraída do terminal
importancias_tree = np.array([0.1550, 0.2380, 0.0000, 0.6068])
norm_tree = normalizar(importancias_tree)

# Altere a média para dividir por 6 frentes
score_consenso = (norm_pearson + norm_mi + norm_mse + norm_kan + norm_var + norm_tree) / 6.0

# 2. Montar o DataFrame Consolidado
df_consenso = pd.DataFrame({
    'Variável': variaveis,
    'Pearson': norm_pearson,
    'Mutual Info': norm_mi,
    'Permutação': norm_mse,
    'Fórmula KAN': norm_kan,
    'Variância': norm_var,
    'SCORE CONSENSO': score_consenso
})

# Ordenar pelo ranking final do Consenso
df_consenso = df_consenso.sort_values(by='SCORE CONSENSO', ascending=False).reset_index(drop=True)

print("\n========================================================")
print("MATRIZ DE CONSENSO MULTICRITÉRIO FINAL (0 a 1):")
print(df_consenso.round(4).to_string(index=False))
print("========================================================")

# 3. Salvar os resultados
df_consenso.round(4).to_csv('super_consenso_ufes.csv', sep=';', index=False)
print("\nSuper Matriz salva com sucesso em 'super_consenso_ufes.csv'!")