import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. CARREGAMENTO E CONFIGURAÇÃO DA JANELA
# ==========================================
# Altere para o nome correto do seu arquivo CSV do INMET
arquivo_csv = "INMET_SE_ES_A612_VITORIA_01-01-2025_A_31-12-2025.CSV" 

df = pd.read_csv(arquivo_csv, sep=';', skiprows=8, encoding='latin-1', decimal=',')

coluna_data = [c for c in df.columns if 'Data' in c or 'DATA' in c][0]
coluna_hora = [c for c in df.columns if 'Hora' in c or 'HORA' in c][0]
df['Data_Hora'] = pd.to_datetime(df[coluna_data] + ' ' + df[coluna_hora], errors='coerce')
df.set_index('Data_Hora', inplace=True)
df = df.replace([-9999, 9999, -9999.0, 9999.0], np.nan)

coluna_vento = [c for c in df.columns if 'VENTO' in c.upper() and 'VELOCIDADE' in c.upper()][0]
df_vento = df[[coluna_vento]].dropna().copy()

# Estruturando a janela temporal (t) para prever (t+1)
X = df_vento[[coluna_vento]].copy()
X.columns = ['vento_atual_(t)']
y = df_vento[coluna_vento].shift(-1)

X = X.iloc[:-1]
y = y.dropna()

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

# Aqui definimos as variáveis que estavam faltando!
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# ==========================================
# 4. CONSTRUÇÃO E TREINO DA REDE NEURAL (MLP)
# ==========================================
modelo_rede_neural = MLPRegressor(
    hidden_layer_sizes=(50, 25), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=42,
    early_stopping=True, 
    validation_fraction=0.15
)

# Agora o fit vai encontrar os dados perfeitamente
modelo_rede_neural.fit(X_train_scaled, y_train_scaled)

# ==========================================
# 5. PREVISÃO E DESNORMALIZAÇÃO
# ==========================================
preds_train_scaled = modelo_rede_neural.predict(X_train_scaled)
preds_val_scaled = modelo_rede_neural.predict(X_val_scaled)
preds_test_scaled = modelo_rede_neural.predict(X_test_scaled)

preds_train = scaler_y.inverse_transform(preds_train_scaled.reshape(-1, 1)).flatten()
preds_val = scaler_y.inverse_transform(preds_val_scaled.reshape(-1, 1)).flatten()
preds_test = scaler_y.inverse_transform(preds_test_scaled.reshape(-1, 1)).flatten()

# ==========================================
# 6. CÁLCULO DOS ERROS
# ==========================================
print("=== Períodos dos Datasets ===")
print(f"Treino:    {X_train.index.min()} até {X_train.index.max()} | Qtd: {len(X_train)}")
print(f"Validação: {X_val.index.min()} até {X_val.index.max()} | Qtd: {len(X_val)}")
print(f"Teste:     {X_test.index.min()} até {X_test.index.max()} | Qtd: {len(X_test)}\n")

def exibir_metricas(y_real, y_pred, label):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    print(f"--- Rede Neural MLP: {label} ---")
    print(f"MAE  (Erro Médio Absoluto): {mae:.4f} m/s")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:.4f} m/s\n")

exibir_metricas(y_train, preds_train, "Treinamento")
exibir_metricas(y_val, preds_val, "Validação")
exibir_metricas(y_test, preds_test, "Teste")