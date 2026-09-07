import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. CARREGAMENTO E CONFIGURAÇÃO DA JANELA (6 HORAS)
# ==========================================
arquivo_csv = "INMET_SE_ES_A612_VITORIA_01-01-2025_A_31-12-2025.CSV" 

df = pd.read_csv(arquivo_csv, sep=';', skiprows=8, encoding='latin-1', decimal=',')

coluna_data = [c for c in df.columns if 'Data' in c or 'DATA' in c][0]
coluna_hora = [c for c in df.columns if 'Hora' in c or 'HORA' in c][0]
df['Data_Hora'] = pd.to_datetime(df[coluna_data] + ' ' + df[coluna_hora], errors='coerce')
df.set_index('Data_Hora', inplace=True)

# 1.1 Substitui os códigos de erro do INMET por NaN
df = df.replace([-9999, 9999, -9999.0, 9999.0], np.nan)

coluna_vento = [c for c in df.columns if 'VENTO' in c.upper() and 'VELOCIDADE' in c.upper()][0]

# 1.2 INTERPOLAÇÃO: Preenche os buracos de forma linear baseando-se no tempo
# limit_direction='both' garante que falhas no extremo início ou fim também sejam tratadas
df_vento = df[[coluna_vento]].copy()
qtd_nulos_antes = df_vento[coluna_vento].isna().sum()

df_vento[coluna_vento] = df_vento[coluna_vento].interpolate(method='linear', limit_direction='both')

print(f"=== Tratamento de Dados ===")
print(f"Valores nulos/falhas encontrados e interpolados: {qtd_nulos_antes}\n")

# 1.3 Criando as colunas do passado (Janela de 6 horas)
df_janela = pd.DataFrame(index=df_vento.index)
for i in range(5, -1, -1):
    nome_col = f'vento_(t-{i})' if i > 0 else 'vento_atual_(t)'
    df_janela[nome_col] = df_vento[coluna_vento].shift(i)

# O Alvo (y) é a próxima hora
df_janela['alvo_(t+1)'] = df_vento[coluna_vento].shift(-1)

# Remove apenas as linhas das bordas extremas geradas pelo shift da janela
df_janela.dropna(inplace=True)

# Separando X e y
X = df_janela.drop(columns=['alvo_(t+1)'])
y = df_janela['alvo_(t+1)']

# ==========================================
# 2. DIVISÃO CRONOLÓGICA (70 / 15 / 15)
# ==========================================
datas = X.index
idx_70 = int(len(datas) * 0.70)
idx_85 = int(len(datas) * 0.85)
data_fim_treino = datas[idx_70]
data_fim_val = datas[idx_85]

X_train = X.loc[X.index < data_fim_treino]
y_train = y.loc[y.index < data_fim_treino]

X_val = X.loc[(X.index >= data_fim_treino) & (X.index < data_fim_val)]
y_val = y.loc[(y.index >= data_fim_treino) & (y.index < data_fim_val)]

X_test = X.loc[X.index >= data_fim_val]
y_test = y.loc[y.index >= data_fim_val]

# ==========================================
# 3. NORMALIZAÇÃO (MinMaxScaler)
# ==========================================
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# ==========================================
# 4. TREINAMENTO E PREVISÃO: XGBOOST
# ==========================================
modelo_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
modelo_xgb.fit(X_train_scaled, y_train_scaled, eval_set=[(X_val_scaled, y_val_scaled)], verbose=False)

preds_test_xgb_scaled = modelo_xgb.predict(X_test_scaled)
preds_test_xgb = scaler_y.inverse_transform(preds_test_xgb_scaled.reshape(-1, 1)).flatten()

# ==========================================
# 5. TREINAMENTO E PREVISÃO: REDE NEURAL (MLP)
# ==========================================
modelo_mlp = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.15)
modelo_mlp.fit(X_train_scaled, y_train_scaled)

preds_test_mlp_scaled = modelo_mlp.predict(X_test_scaled)
preds_test_mlp = scaler_y.inverse_transform(preds_test_mlp_scaled.reshape(-1, 1)).flatten()

# ==========================================
# 6. CÁLCULO E COMPARAÇÃO DOS ERROS NO TESTE
# ==========================================
def calcular_erros(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    return mae, rmse

mae_xgb, rmse_xgb = calcular_erros(y_test, preds_test_xgb)
mae_mlp, rmse_mlp = calcular_erros(y_test, preds_test_mlp)

tabela_comparativa = pd.DataFrame({
    'Métrica': ['MAE (Teste)', 'RMSE (Teste)'],
    'XGBoost (6h)': [mae_xgb, rmse_xgb],
    'Rede Neural MLP (6h)': [mae_mlp, rmse_mlp]
})

print("\n==========================================")
print("   COMPARAÇÃO COM JANELA TEMPORAL DE 6H   ")
print("==========================================")
print(tabela_comparativa.to_string(index=False))


import matplotlib.pyplot as plt

# ==========================================
# 7. PLOTAR A CURVA DE TREINAMENTO (LOSS CURVE)
# ==========================================
plt.figure(figsize=(10, 5))

# Plota a curva de perda do treino
plt.plot(modelo_mlp.loss_curve_, label='Erro de Treinamento (Loss)', color='blue', linewidth=2)

# Se o early stopping estiver ativo, o scikit-learn também guarda a perda da validação
if hasattr(modelo_mlp, 'validation_scores_'):
    # Convertendo o score de validação (R²) de volta para uma aproximação de perda
    perda_val = [1 - score for score in modelo_mlp.validation_scores_]
    plt.plot(perda_val, label='Erro de Validação', color='red', linestyle='--', linewidth=2)

plt.title('Curva de Aprendizado da Rede Neural (Janela de 6h)', fontsize=14, fontweight='bold')
plt.xlabel('Épocas (Iterações)', fontsize=12)
plt.ylabel('Erro Quadrático Médio (MSE)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

# Salva o gráfico automaticamente como imagem para usar no seu relatório
plt.savefig('curva_treinamento_mlp.png', dpi=300, bbox_inches='tight')
print("Gráfico da curva de treinamento salvo com sucesso como 'curva_treinamento_mlp.png'!")
plt.show()