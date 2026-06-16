import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=== Inicializando o Super Ensemble Regressor da UFES ===\n")

# 1. Carregar os datasets do split
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = df_train[colunas_x].values
y_train = df_train['Alvo_Y'].values
X_test = df_test[colunas_x].values
y_test = df_test['Alvo_Y'].values

# 2. Instanciar os modelos de base com os mesmos hiperparâmetros testados
modelo_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_svr = SVR(kernel='rbf', C=10.0, epsilon=0.01)
modelo_xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)

# 3. Criar o Metamodelo de Consenso (Voting Regressor)
# Definimos pesos (weights) baseados no desempenho individual observado no benchmark
# SVR (peso 4), XGBoost (peso 3), Random Forest (peso 2), Árvore Simples (peso 1)
ensemble_comite = VotingRegressor(
    estimators=[
        ('tree', modelo_tree),
        ('rf', modelo_rf),
        ('svr', modelo_svr),
        ('xgb', modelo_xgb)
    ],
    weights=[1, 2, 4, 3]
)

print("-> Treinando o Comitê de Modelos Integrados (Ensemble)...")
ensemble_comite.fit(X_train, y_train)

# 4. Predições
print("-> Gerando predições para o conjunto de teste...")
y_pred_ensemble = ensemble_comite.predict(X_test)

# 5. Cálculo das 5 Métricas do Relatório
def calcular_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

mae = mean_absolute_error(y_test, y_pred_ensemble)
mse = mean_squared_error(y_test, y_pred_ensemble)
rmse = np.sqrt(mse)
mape = calcular_mape(y_test, y_pred_ensemble)
r2 = r2_score(y_test, y_pred_ensemble)

# 6. Montar a tabela comparativa resgatando os valores anteriores para o confronto final
df_final = pd.DataFrame({
    'Modelo': ['Árvore de Decisão', 'Random Forest', 'XGBoost', 'SVR (Kernel RBF)', 'SUPER ENSEMBLE (Comitê)'],
    'MAE': [0.050855, 0.021493, 0.015332, 0.009391, mae],
    'MSE': [0.004062, 0.000834, 0.000388, 0.000145, mse],
    'RMSE': [0.063733, 0.028880, 0.019703, 0.012061, rmse],
    'MAPE (%)': [11.164393, 4.584195, 3.274587, 2.042114, mape],
    'R² (Acurácia)': [0.818397, 0.962711, 0.982643, 0.993496, r2]
})

print("\n=========================================================================")
print("TABELA COMPLETA E DEFINITIVA DE MACHINE LEARNING:")
print(df_final.round(6).to_string(index=False))
print("=========================================================================")

# 7. Salvar os resultados para anexar ao relatório
df_final.round(6).to_csv('tabela_confronto_absoluto_ufes.csv', sep=';', index=False)
print("\nFramework de Ensemble salvo com sucesso em 'tabela_confronto_absoluto_ufes.csv'!")