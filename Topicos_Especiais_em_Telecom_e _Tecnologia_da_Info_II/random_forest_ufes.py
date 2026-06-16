import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Iniciando o treinamento e comparação dos modelos...")

# 1. Carregar os mesmos datasets do split
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

# Modelo A: Árvore de Decisão Simples (Profundidade Limitada)
tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

# Modelo B: Random Forest (Ensemble com 100 árvores)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# ==========================================
# AVALIAÇÃO DE DESEMPENHO
# ==========================================
y_pred_tree = tree_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

metrics = {
    'Modelo': ['Árvore de Decisão Simples', 'Random Forest (Ensemble)'],
    'MSE (Erro)': [mean_squared_error(y_test, y_pred_tree), mean_squared_error(y_test, y_pred_rf)],
    'R² (Acurácia)': [r2_score(y_test, y_pred_tree), r2_score(y_test, y_pred_rf)]
}
df_metrics = pd.DataFrame(metrics)

# ==========================================
# COMPARAÇÃO DE IMPORTÂNCIA DAS VARIÁVEIS
# ==========================================
df_importancias = pd.DataFrame({
    'Variável': colunas_x,
    'Importância Árvore': tree_model.feature_importances_,
    'Importância Random Forest': rf_model.feature_importances_
}).sort_values(by='Importância Random Forest', ascending=False)

# Print das Tabelas no Terminal
print("\n========================================================")
print("TABELA COMPARATIVA DE DESEMPENHO:")
print(df_metrics.to_string(index=False))
print("========================================================")

print("\n========================================================")
print("COMPARAÇÃO DE IMPORTÂNCIA DAS VARIÁVEIS:")
print(df_importancias.to_string(index=False))
print("========================================================")

# ==========================================
# PLOT GRÁFICO DO COMPARATIVO DE IMPORTÂNCIA
# ==========================================
x_indices = np.arange(len(colunas_x))
largura_barra = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x_indices - largura_barra/2, df_importancias['Importância Árvore'], largura_barra, label='Árvore Simples', color='#b3cde3')
plt.bar(x_indices + largura_barra/2, df_importancias['Importância Random Forest'], largura_barra, label='Random Forest', color='#2ca25f')

plt.xlabel('Variáveis Independentes')
plt.ylabel('Score de Importância (MDI)')
plt.title('Comparativo de Importância: Árvore Simples vs. Random Forest')
plt.xticks(x_indices, df_importancias['Variável'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig('comparativo_modelos_tree_rf.png', bbox_inches='tight', dpi=300)
print("\nGráfico comparativo salvo com sucesso em 'comparativo_modelos_tree_rf.png'!")