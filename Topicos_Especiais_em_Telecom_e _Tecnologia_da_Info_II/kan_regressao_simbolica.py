import pandas as pd
import numpy as np
import torch
from kan import KAN
import matplotlib.pyplot as plt
import sympy as sp

# 1. Carregar os dados normalizados
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

# O pykan precisa dos dados no formato de tensores do PyTorch
X_train = torch.tensor(df_train[['Var_A', 'Var_B', 'Var_C', 'Var_D']].values, dtype=torch.float32)
y_train = torch.tensor(df_train['Alvo_Y'].values, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(df_test[['Var_A', 'Var_B', 'Var_C', 'Var_D']].values, dtype=torch.float32)
y_test = torch.tensor(df_test['Alvo_Y'].values, dtype=torch.float32).unsqueeze(1)

# Organizar em um dicionário que a KAN espera
dataset = {
    'train_input': X_train,
    'train_label': y_train,
    'test_input': X_test,
    'test_label': y_test
}

# 2. Criar a arquitetura KAN
# [4, 10, 1] significa: 4 entradas, 1 camada escondida de 10, 1 saída
# k=3 é o grau do polinômio B-spline (padrão)
# grid=5 é o número de divisões da curva spline
model = KAN(width=[4, 10, 1], grid=5, k=3, seed=42)

# 3. Treinar a KAN
print("Treinando a KAN (Kolmogorov-Arnold Network)...")
# Ajustado: mudou 'opt' para 'optimizer' e adicionou 'lr=1.0' (comum para LBFGS no pykan)
model.fit(dataset, opt="LBFGS", steps=50, lr=1.0) 
print("Treinamento concluído!")

# 4. Fazer previsões no teste
# Como o PyTorch calcula gradientes automaticamente, precisamos desativá-los para extrair os números puros (.numpy())
with torch.no_grad():
    y_pred_kan = model(X_test).numpy()

y_test_np = y_test.numpy()

# 5. Gerar o gráfico de comparação para ver o resultado
plt.figure(figsize=(6, 6))
plt.scatter(y_test_np, y_pred_kan, color='purple', alpha=0.6, label='Previsões KAN')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfeito (Y = X)')
plt.title('Valor Real vs. Previsão da KAN')
plt.xlabel('Valor Real (Normalizado)')
plt.ylabel('Previsão da KAN (Normalizado)')
plt.legend()
plt.grid(True)

plt.savefig('resultado_kan.png')
print("Gráfico 'resultado_kan.png' salvo com sucesso!")

# 6. Regressão Simbólica Controlada (Sem zerar a rede)
print("\nConvertendo as curvas numéricas da KAN em símbolos matemáticos...")

# Definimos as funções permitidas para evitar que o algoritmo desista e zere as conexões
biblioteca_funcoes = ['x', 'x^2', 'sin', 'cos']

try:
    # O auto_symbolic converte as curvas usando apenas a nossa biblioteca alvo
    model.auto_symbolic(lib=biblioteca_funcoes)
except Exception as e:
    print(f"Nota no ajuste simbólico: {e}")

print("\nExtraindo a fórmula final consolidada...")
var_entrada = sp.symbols('Var_A Var_B Var_C Var_D')

# Extrai a equação final mapeada para as variáveis da UFES
formulas = model.symbolic_formula(var=var_entrada)

print("\n========================================================")
print("EQUAÇÃO SIMBÓLICA FINAL EXTRAÍDA DA KAN:")
print(formulas[0][0])
print("========================================================")

# 7. Gerar e salvar o gráfico computacional com as equações
print("\nGerando o gráfico computacional com as fórmulas aplicadas...")
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 9))
    
    # O método plot vai desenhar o grafo mostrando os nós e as funções fixadas (cos, x^2, x)
    model.plot(beta=100) 
    
    plt.savefig('mapa_regressao_simbolica.png', bbox_inches='tight')
    print("Gráfico computacional salvo com sucesso em 'mapa_regressao_simbolica.png'!")
except Exception as e:
    print(f"Erro ao gerar o gráfico computacional: {e}")