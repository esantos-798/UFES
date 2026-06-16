import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Carregar os dados de treinamento e validação
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_val = pd.read_csv('dataset_validacao_split.csv')

# Separar entradas (X) e saídas (y)
X_train = df_train[['Var_A', 'Var_B', 'Var_C', 'Var_D']]
y_train = df_train['Alvo_Y']

X_val = df_val[['Var_A', 'Var_B', 'Var_C', 'Var_D']]
y_val = df_val['Alvo_Y']

# 2. Criar a Rede Neural (Multi-Layer Perceptron)
# hidden_layer_sizes=(10,) significa 1 camada escondida com 10 neurônios
# activation='relu' é a mesma função de ativação
# max_iter=5000 é o número máximo de épocas de treino
model = MLPRegressor(
    #hidden_layer_sizes=(10,), 
    hidden_layer_sizes=(10, 10), 
    activation='relu', 
    #solver='adam', 
    solver='lbfgs',
    random_state=42,
    max_iter=5000,
    verbose=True
)

# 3. Treinar a Rede Neural
print("Treinando a rede neural...")
model.fit(X_train, y_train)
print("Treinamento concluído!")

# 4. Avaliar o modelo com os dados de validação
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print("\n--- Resultados na Validação ---")
print(f"Erro Quadrático Médio (MSE): {mse:.6f}")
print(f"Erro Médio Absoluto (MAE): {mae:.6f}")

# 5. Avaliação Definitiva com o Conjunto de Teste (Dados Inéditos)
df_test = pd.read_csv('dataset_teste_split.csv')

X_test = df_test[['Var_A', 'Var_B', 'Var_C', 'Var_D']]
y_test = df_test['Alvo_Y']

# Fazendo as previsões finais
y_pred_test = model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("\n--- TESTE FINAL (Dados Inéditos) ---")
print(f"Erro Quadrático Médio (MSE): {mse_test:.6f}")
print(f"Erro Médio Absoluto (MAE): {mae_test:.6f}")

# Exibir os 5 primeiros resultados comparando o Real vs Previsto
print("\nComparação Prática (Primeiras 5 amostras):")
resultado = pd.DataFrame({'Valor Real': y_test, 'Previsão da Rede': y_pred_test})
print(resultado.head())

# Criando o gráfico de dispersão
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Previsões')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Perfeito (Y = X)')

plt.title('Valor Real vs. Previsão da Rede Neural')
plt.xlabel('Valor Real (Normalizado)')
plt.ylabel('Previsão da Rede (Normalizado)')
plt.legend()
plt.grid(True)

# Salva o gráfico na sua pasta
plt.savefig('resultado_regressao.png')
print("\nGráfico 'resultado_regressao.png' salvo com sucesso!")