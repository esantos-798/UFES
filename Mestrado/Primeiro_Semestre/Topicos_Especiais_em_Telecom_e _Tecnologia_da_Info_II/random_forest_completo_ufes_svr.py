import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Treinando a Support Vector Regression (SVR)...")

# 1. Carregar dados
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = df_train[colunas_x].values
y_train = df_train['Alvo_Y'].values
X_test = df_test[colunas_x].values
y_test = df_test['Alvo_Y'].values

# DICA TÉCNICA: SVR é extremamente sensível à escala. Como seus dados já 
# estão normalizados, o RBF vai funcionar muito bem.

# 2. Instanciar SVR com Kernel RBF (C e epsilon controlam a rigidez do tubo)
scaler_y = StandardScaler() # Opcional: padronizar o alvo pode ajudar o SVR
modelo_svr = SVR(kernel='rbf', C=10.0, epsilon=0.01)
modelo_svr.fit(X_train, y_train)

# 3. Predição e Métricas
y_pred_svr = modelo_svr.predict(X_test)

def calcular_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

print("\n========================================================")
print("DESEMPENHO DA SVR:")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_svr):.6f}")
print(f"MSE:  {mean_squared_error(y_test, y_pred_svr):.6f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.6f}")
print(f"MAPE: {calcular_mape(y_test, y_pred_svr):.4f}%")
print(f"R²:   {r2_score(y_test, y_pred_svr):.4f}")
print("========================================================")

# 4. Função customizada para o MAPE (com tratamento para evitar divisão por zero)
def calcular_mape(y_true, y_pred):
    epsilon = 1e-5
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


# 5. Cálculo das Métricas para cada modelo
metrics = {
    'Modelo': ['Árvore de Decisão Simples', 'Random Forest (Ensemble)'],
    'MAE': [mean_absolute_error(y_test, y_pred_tree), mean_absolute_error(y_test, y_pred_rf)],
    'MSE': [mean_squared_error(y_test, y_pred_tree), mean_squared_error(y_test, y_pred_rf)],
    'RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_tree)), np.sqrt(mean_squared_error(y_test, y_pred_rf))],
    'MAPE (%)': [calcular_mape(y_test, y_pred_tree), calcular_mape(y_test, y_pred_rf)],
    'R² (Acurácia)': [r2_score(y_test, y_pred_tree), r2_score(y_test, y_pred_rf)]
}

df_metrics = pd.DataFrame(metrics)

print("\n=========================================================================")
print("TABELA COMPARATIVA DE DESEMPENHO ESTENDIDA:")
print(df_metrics.round(6).to_string(index=False))
print("=========================================================================")

# Salvar a tabela de métricas estendida
df_metrics.round(6).to_csv('comparativo_metricas_estendidas.csv', sep=';', index=False)
print("\nMétricas salvas com sucesso em 'comparativo_metricas_estendidas.csv'!")