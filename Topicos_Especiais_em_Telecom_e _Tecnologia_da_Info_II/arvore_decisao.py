import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

print("Treinando a Árvore de Decisão...")

# 1. Carregar os mesmos datasets do split
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = df_train[colunas_x].values
y_train = df_train['Alvo_Y'].values
X_test = df_test[colunas_x].values
y_test = df_test['Alvo_Y'].values

# 2. Instanciar e treinar o modelo
# Limitamos max_depth para a árvore não gigantear e podermos plotar com clareza
modelo_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
modelo_tree.fit(X_train, y_train)

# 3. Avaliar o modelo
y_pred = modelo_tree.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"-> Desempenho da Árvore | MSE: {mse:.6f} | R²: {r2:.4f}")

# 4. Extrair a Importância das Variáveis (Gini/Variance Reduction)
importancias_tree = modelo_tree.feature_importances_

df_importancia = pd.DataFrame({
    'Variável': colunas_x,
    'Importância na Árvore': importancias_tree
}).sort_values(by='Importância na Árvore', ascending=False)

print("\n========================================================")
print("IMPORTÂNCIA DAS VARIÁVEIS PELA ÁRVORE DE DECISÃO:")
print(df_importancia.to_string(index=False))
print("========================================================")

# 5. Exportar gráfico da estrutura da Árvore
plt.figure(figsize=(20, 10))
plot_tree(modelo_tree, feature_names=colunas_x, filled=True, rounded=True, fontsize=10)
plt.savefig('estrutura_arvore_decisao.png', bbox_inches='tight', dpi=300)
print("\nGráfico da estrutura salvo em 'estrutura_arvore_decisao.png'!")

# 6. Salvar as importâncias para usarmos no seu Super Consenso se quiser
df_importancia.to_csv('importancia_arvore.csv', sep=';', index=False)