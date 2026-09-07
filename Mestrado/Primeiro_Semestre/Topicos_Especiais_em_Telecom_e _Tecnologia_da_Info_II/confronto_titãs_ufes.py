import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Iniciando o Confronto Definitivo de Titãs (Árvore vs RF vs SVR vs XGBoost)...")

# 1. Carregar os datasets do split
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = df_train[colunas_x].values
y_train = df_train['Alvo_Y'].values
X_test = df_test[colunas_x].values
y_test = df_test['Alvo_Y'].values

# ==========================================
# TREINAMENTO DOS MODELOS
# ==========================================
print("-> Treinando Árvore de Decisão Simples...")
tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

print("-> Treinando Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("-> Treinando Support Vector Regression...")
svr_model = SVR(kernel='rbf', C=10.0, epsilon=0.01)
svr_model.fit(X_train, y_train)

print("-> Treinando XGBoost Regressor...")
# n_estimators=100 e max_depth=4 para manter um paralelo justo com as outras estruturas
xgb_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# ==========================================
# PREDIÇÕES
# ==========================================
y_pred_tree = tree_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_svr = svr_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# ==========================================
# CÁLCULO DAS MÉTRICAS
# ==========================================
def calcular_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

modelos_nomes = ['Árvore de Decisão', 'Random Forest', 'SVR (Kernel RBF)', 'XGBoost']
preds = [y_pred_tree, y_pred_rf, y_pred_svr, y_pred_xgb]

maes = [mean_absolute_error(y_test, p) for p in preds]
mses = [mean_squared_error(y_test, p) for p in preds]
rmses = [np.sqrt(m) for m in mses]
mapes = [calcular_mape(y_test, p) for p in preds]
r2s = [r2_score(y_test, p) for p in preds]

# 2. Montar Tabela Comparativa Final
df_confronto = pd.DataFrame({
    'Modelo': modelos_nomes,
    'MAE': maes,
    'MSE': mses,
    'RMSE': rmses,
    'MAPE (%)': mapes,
    'R² (Acurácia)': r2s
})

print("\n=========================================================================")
print("TABELA COMPARATIVA DE DESEMPENHO ATUALIZADA (COM XGBOOST):")
print(df_confronto.round(6).to_string(index=False))
print("=========================================================================")

# 3. Salvar os resultados para o relatório
df_confronto.round(6).to_csv('tabela_confronto_titas_ufes.csv', sep=';', index=False)
print("\nTabela final de titãs salva com sucesso em 'tabela_confronto_titas_ufes.csv'!")