import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from kan import MultKAN  # Importa a estrutura da KAN

# 1. Configurar reprodutibilidade e carregar dados de teste
np.random.seed(42)
df_test = pd.read_csv('dataset_teste_split.csv')

X_test_np = df_test[['Var_A', 'Var_B', 'Var_C', 'Var_D']].values
y_test_np = df_test['Alvo_Y'].values

# Converter para Tensores do PyTorch
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)

# RECONSTRUIR E CARREGAR O MODELO TREINADO:
# Usamos a mesma arquitetura que gerou o seu gráfico (4 entradas, 10 neurônios ocultos, 1 saída)
model = MultKAN(width=[4, 10, 1], grid=5, k=3) 

try:
    # Carrega a versão estável após o treinamento (v1.0 ou v0.1 dependendo do último save)
    model.load_ckpt('0.1') 
    print("Modelo KAN carregado com sucesso do checkpoint v0.1!")
except Exception:
    try:
        model.load_ckpt('0.2')
        print("Modelo KAN carregado com sucesso do checkpoint v0.2!")
    except Exception as e:
        print(f"Não foi possível carregar o checkpoint automaticamente: {e}")
        print("Dica: Você também pode colar o código de ranking direto no final do seu script principal.")

# 2. Fazer a previsão base (sem alterações)
with torch.no_grad():
    y_pred_base = model(X_test_tensor).numpy()

erro_base = mean_squared_error(y_test_np, y_pred_base)

# 3. Algoritmo de Importância por Permutação
nomes_variaveis = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
importancias = []

print("Calculando o impacto de cada variável no modelo KAN...")

for idx, nome in enumerate(nomes_variaveis):
    # Criar uma cópia dos dados de teste
    X_test_permutado = X_test_np.copy()
    
    # Embaralhar (permutar) apenas a coluna da variável atual
    np.random.shuffle(X_test_permutado[:, idx])
    
    # Passar os dados alterados pelo modelo KAN
    X_permutado_tensor = torch.tensor(X_test_permutado, dtype=torch.float32)
    with torch.no_grad():
        y_pred_permutado = model(X_permutado_tensor).numpy()
    
    # Calcular o novo erro (MSE)
    erro_permutado = mean_squared_error(y_test_np, y_pred_permutado)
    
    # A importância é o quanto o erro aumentou (Aumento do MSE)
    impacto = erro_permutado - erro_base
    importancias.append(impacto)
    print(f"-> {nome} desconfigurada | Aumento no Erro (MSE): {impacto:.6f}")

# 4. Criar o DataFrame do Ranking
df_ranking = pd.DataFrame({
    'Variável': nomes_variaveis,
    'Impacto no Erro (MSE)': importancias
})

# Ordenar do maior impacto (mais importante) para o menor
df_ranking = df_ranking.sort_values(by='Impacto no Erro (MSE)', ascending=False).reset_index(drop=True)

print("\n========================================================")
print("RANKING OFICIAL DE INFLUÊNCIA DA KAN:")
print(df_ranking)
print("========================================================")

# 5. Gerar Gráfico de Barras do Ranking
plt.figure(figsize=(8, 5))
cores = ['#7bc043' if idx == 0 else '#d3d3d3' for idx in range(len(df_ranking))] # Destaca a campeã
plt.barh(df_ranking['Variável'][::-1], df_ranking['Impacto no Erro (MSE)'][::-1], color=cores)
plt.xlabel('Aumento no Erro do Modelo (MSE) ao remover a variável')
plt.title('Importância das Variáveis de Decisão na KAN')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.savefig('ranking_features_kan.png', bbox_inches='tight')
print("\nGráfico de ranking salvo com sucesso em 'ranking_features_kan.png'!")