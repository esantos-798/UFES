import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from kan import KAN

print("=== Inicializando o Treinamento da KAN (Kolmogorov-Arnold Network) ===")

# 1. Carregar os mesmos datasets do split
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']

X_train = torch.tensor(df_train[colunas_x].values, dtype=torch.float32)
y_train = torch.tensor(df_train['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(df_test[colunas_x].values, dtype=torch.float32)
y_test = torch.tensor(df_test['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)

dataset = {
    'train_input': X_train,
    'train_label': y_train,
    'test_input': X_test,
    'test_label': y_test
}

# 2. Criar a arquitetura da KAN [4, 2, 1]
modelo_kan = KAN(width=[4, 2, 1], grid=5, k=3, seed=42)

# 3. Treinar a rede (Sintaxe atualizada para pykan)
print("\n-> Treinando o modelo...")
# Nas versões novas, passamos 'optimizer' ou deixamos o LBFGS agir por padrão
resultados = modelo_kan.train(dataset, opt="LBFGS", steps=20) if hasattr(modelo_kan.train, 'opt') else modelo_kan.fit(dataset, steps=20, opt="LBFGS")
# Nota: Se mesmo assim a assinatura reclamar, o fallback abaixo resolve usando a chamada limpa padrão:
if 'TypeError' in locals() or True:
    try:
        # Tenta a sintaxe limpa mais comum das versões estáveis de 2024-2026:
        resultados = modelo_kan.train(dataset, steps=20, opt="LBFGS")
    except TypeError:
        # Fallback para a API simplificada baseada em dicionários de otimização
        resultados = modelo_kan.fit(dataset, steps=20)

# 4. Avaliação de Desempenho
with torch.no_grad():
    y_pred = modelo_kan(X_test)
    mse_kan = torch.nn.functional.mse_loss(y_pred, y_test).item()
    rmse_kan = np.sqrt(mse_kan)
    mae_kan = torch.mean(torch.abs(y_pred - y_test)).item()
    
    y_test_np = y_test.numpy()
    y_pred_np = y_pred.numpy()
    ss_res = np.sum((y_test_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_test_np - np.mean(y_test_np)) ** 2)
    r2_kan = 1 - (ss_res / ss_tot)
    
    mape_kan = np.mean(np.abs((y_test_np - y_pred_np) / (y_test_np + 1e-5))) * 100

print("\n========================================================")
print("DESEMPENHO DA KAN NO CONJUNTO DE TESTE:")
print(f"MAE:  {mae_kan:.6f}")
print(f"MSE:  {mse_kan:.6f}")
print(f"RMSE: {rmse_kan:.6f}")
print(f"MAPE: {mape_kan:.4f}%")
print(f"R²:   {r2_kan:.4f}")
print("========================================================")

# 5. Plotar e salvar a estrutura simbólica da rede
print("\n-> Gerando o gráfico da topologia simbólica...")
plt.figure(figsize=(10, 8))
modelo_kan.plot(beta=10)
plt.savefig('topologia_simbolica_kan.png', bbox_inches='tight', dpi=300)
print("Gráfico da topologia salvo em 'topologia_simbolica_kan.png'!")