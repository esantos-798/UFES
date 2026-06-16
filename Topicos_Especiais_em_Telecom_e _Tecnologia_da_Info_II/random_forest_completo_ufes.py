import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Iniciando o treinamento e cálculo das métricas estendidas...")

# 1. Carregar os datasets do split
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = df_train[colunas_x].values
y_train = df_train['Alvo_Y'].values
X_test = df_test[colunas_x].values
y_test = df_test['Alvo_Y'].values

# 2. Treinamento dos Modelos
tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 3. Predições para o conjunto de teste
y_pred_tree = tree_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

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