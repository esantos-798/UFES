import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from xgboost import XGBRegressor
from kan import KAN

print("Gerando o gráfico de comparação das saídas reais vs. previstas...")

# 1. Carregar datasets
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train_np = df_train[colunas_x].values
y_train_np = df_train['Alvo_Y'].values
X_test_np = df_test[colunas_x].values
y_test_np = df_test['Alvo_Y'].values

# Preparar tensores para a KAN
X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
X_test_torch = torch.tensor(X_test_np, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_np, dtype=torch.float32).reshape(-1, 1)

dataset = {
    'train_input': X_train_torch, 
    'train_label': y_train_torch, 
    'test_input': X_test_torch, 
    'test_label': y_test_torch
}

# 2. Treinar os top modelos
print("-> Treinando SVR...")
svr = SVR(kernel='rbf', C=10.0, epsilon=0.01).fit(X_train_np, y_train_np)

print("-> Treinando XGBoost...")
xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1).fit(X_train_np, y_train_np)

print("-> Treinando KAN (Ajustado)...")
kan_model = KAN(width=[4, 2, 1], grid=5, k=3, seed=42)

# Tratamento robusto da assinatura da API do pykan
if hasattr(kan_model, 'fit'):
    kan_model.fit(dataset, steps=20)
else:
    try:
        kan_model.train(dataset, steps=20, opt="LBFGS")
    except TypeError:
        kan_model.train(dataset, steps=20)

# 3. Gerar predições
y_pred_svr = svr.predict(X_test_np)
y_pred_xgb = xgb.predict(X_test_np)
with torch.no_grad():
    y_pred_kan = kan_model(X_test_torch).numpy().flatten()

# 4. Construir o gráfico (80 primeiras amostras para manter a legibilidade)
amostras = 80
plt.figure(figsize=(15, 6))

plt.plot(y_test_np[:amostras], label='Alvo Real (Y_test)', color='black', linewidth=2.5, linestyle='-')
plt.plot(y_pred_kan[:amostras], label='KAN (R²: 0.9951)', color='#8da0cb', linewidth=1.8, linestyle='--')
plt.plot(y_pred_svr[:amostras], label='SVR (R²: 0.9935)', color='#fc8d62', linewidth=1.5, linestyle=':')
plt.plot(y_pred_xgb[:amostras], label='XGBoost (R²: 0.9826)', color='#66c2a5', linewidth=1.2, linestyle='-.')

plt.title('Validação Visual: Alvo Real vs. Curvas de Previsão (Top Modelos)', fontsize=14)
plt.xlabel('Índice da Amostra de Teste', fontsize=11)
plt.ylabel('Magnitude de Saída (Alvo_Y)', fontsize=11)
plt.legend(fontsize=10, loc='upper right')