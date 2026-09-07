import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. CARREGAMENTO E INTERPOLAÇÃO MULTIVARIADA
# ==========================================
arquivo_csv = "INMET_SE_ES_A612_VITORIA_01-01-2025_A_31-12-2025.CSV" 

df = pd.read_csv(arquivo_csv, sep=';', skiprows=8, encoding='latin-1', decimal=',')

coluna_data = [c for c in df.columns if 'Data' in c or 'DATA' in c][0]
coluna_hora = [c for c in df.columns if 'Hora' in c or 'HORA' in c][0]
df['Data_Hora'] = pd.to_datetime(df[coluna_data] + ' ' + df[coluna_hora], errors='coerce')
df.set_index('Data_Hora', inplace=True)
df = df.replace([-9999, 9999, -9999.0, 9999.0], np.nan)

# Identificando as 3 colunas dinamicamente
col_vento = [c for c in df.columns if 'VENTO' in c.upper() and 'VELOCIDADE' in c.upper()][0]
col_temp = [c for c in df.columns if 'TEMPERATURA' in c.upper() and 'AR' in c.upper() and 'BULBO' in c.upper()][0] # Ajuste se necessário
col_pressao = [c for c in df.columns if 'PRESSAO' in c.upper() and 'ESTACAO' in c.upper()][0] # Ajuste se necessário

# Criando dataframe com as 3 features e aplicando interpolação linear em todas
df_features = df[[col_vento, col_temp, col_pressao]].copy()
df_features = df_features.interpolate(method='linear', limit_direction='both')

# ==========================================
# 2. CRIAÇÃO DA JANELA TEMPORAL DE 6 HORAS (MULTIVARIADO)
# ==========================================
df_janela = pd.DataFrame(index=df_features.index)

# Para cada uma das 3 variáveis, criamos os 6 lags do passado
for col in df_features.columns:
    var_nome = 'vento' if col == col_vento else ('temp' if col == col_temp else 'pressao')
    for i in range(5, -1, -1):
        nome_lag = f'{var_nome}_(t-{i})' if i > 0 else f'{var_nome}_atual_(t)'
        df_janela[nome_lag] = df_features[col].shift(i)

# O Alvo (y) continua sendo APENAS a velocidade do vento na próxima hora (t+1)
df_janela['alvo_(t+1)'] = df_features[col_vento].shift(-1)
df_janela.dropna(inplace=True)

# Separando X (Agora com 18 colunas: 3 variáveis x 6 horas) e y
X = df_janela.drop(columns=['alvo_(t+1)'])
y = df_janela['alvo_(t+1)']

# ==========================================
# 3. DIVISÃO CRONOLÓGICA (70 / 15 / 15)
# ==========================================
datas = X.index
idx_70 = int(len(datas) * 0.70)
idx_85 = int(len(datas) * 0.85)

X_train, y_train = X.loc[X.index < datas[idx_70]], y.loc[y.index < datas[idx_70]]
X_val, y_val = X.loc[(X.index >= datas[idx_70]) & (X.index < datas[idx_85])], y.loc[(y.index >= datas[idx_70]) & (y.index < datas[idx_85])]
X_test, y_test = X.loc[X.index >= datas[idx_85]], y.loc[y.index >= datas[idx_85]]

# Normalização
scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
X_train_s = scaler_X.fit_transform(X_train)
y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
X_val_s = scaler_X.transform(X_val)
y_val_s = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
X_test_s = scaler_X.transform(X_test)
y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Formatação 3D para as Redes Recorrentes (PyTorch) -> [Amostras, Timesteps (6), Features (3)]
X_train_3D = torch.tensor(X_train_s.reshape(-1, 6, 3), dtype=torch.float32)
y_train_3D = torch.tensor(y_train_s.reshape(-1, 1), dtype=torch.float32)
X_val_3D = torch.tensor(X_val_s.reshape(-1, 6, 3), dtype=torch.float32)
y_val_3D = torch.tensor(y_val_s.reshape(-1, 1), dtype=torch.float32)
X_test_3D = torch.tensor(X_test_s.reshape(-1, 6, 3), dtype=torch.float32)

# Dicionário para armazenar os erros finais
resultados = {}

def calcular_erros(y_real, y_pred):
    return mean_absolute_error(y_real, y_pred), np.sqrt(mean_squared_error(y_real, y_pred))

# ==========================================
# 4. EXECUÇÃO: MODELOS CLÁSSICOS (MLP E XGBOOST)
# ==========================================
print("Treinando MLP Multivariada...")
mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, early_stopping=True)
mlp.fit(X_train_s, y_train_s)
preds_mlp = scaler_y.inverse_transform(mlp.predict(X_test_s).reshape(-1, 1)).flatten()
resultados['Rede Neural MLP (Multivariada)'] = calcular_erros(y_test, preds_mlp)

print("Treinando XGBoost Multivariado...")
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=42)
xgb.fit(X_train_s, y_train_s, eval_set=[(X_val_s, y_val_s)], verbose=False)
preds_xgb = scaler_y.inverse_transform(xgb.predict(X_test_s).reshape(-1, 1)).flatten()
resultados['XGBoost (Multivariado)'] = calcular_erros(y_test, preds_xgb)

# ==========================================
# 5. EXECUÇÃO RECORRENTE COLETANDO HISTÓRICO
# ==========================================
class MultivariadoLSTM(nn.Module):
    def __init__(self, bidirecional=False):
        super(MultivariadoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=16, num_layers=1, batch_first=True, bidirectional=bidirecional)
        self.linear = nn.Linear(16 * 2 if bidirecional else 16, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

def treinar_com_historico(bidirecional, nome_chave):
    modelo = MultivariadoLSTM(bidirecional=bidirecional)
    criterio = nn.MSELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.005, weight_decay=1e-4)
    
    historico = []
    for epoca in range(60):
        modelo.train()
        otimizador.zero_grad()
        loss = criterio(modelo(X_train_3D), y_train_3D)
        l1_reg = sum(torch.sum(torch.abs(p)) for p in modelo.parameters())
        loss_total = loss + 1e-5 * l1_reg
        loss_total.backward()
        otimizador.step()
        historico.append(loss.item()) # Armazena o MSE puro de treino
        
    modelo.eval()
    with torch.no_grad():
        preds_s = modelo(X_test_3D).numpy()
    preds = scaler_y.inverse_transform(preds_s).flatten()
    resultados[nome_chave] = calcular_erros(y_test, preds)
    return historico

print("Treinando e coletando curvas...")
hist_lstm = treinar_com_historico(bidirecional=False, nome_chave='LSTM 16 cel (Multivariada + L1/L2)')
hist_bilstm = treinar_com_historico(bidirecional=True, nome_chave='BiLSTM 16 cel (Multivariada + L1/L2)')

# ==========================================
# 6. PLOTAGEM UNIFICADA DAS CURVAS DE TREINO
# ==========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# 1. Curva da MLP (Limitamos ao tamanho das épocas executadas ou usamos o tamanho real dela)
plt.plot(mlp.loss_curve_, label='MLP Multivariada (Campeã)', color='blue', linewidth=2.5)

# 2. Curva da LSTM
plt.plot(hist_lstm, label='LSTM 16 cel (Multivariada)', color='crimson', linestyle='--', linewidth=2)

# 3. Curva da BiLSTM
plt.plot(hist_bilstm, label='BiLSTM 16 cel (Multivariada)', color='teal', linestyle=':', linewidth=2)

plt.title('Análise de Convergência: MLP vs LSTM vs BiLSTM (Multivariado)', fontsize=14, fontweight='bold')
plt.xlabel('Épocas de Treinamento', fontsize=12)
plt.ylabel('Erro Quadrático Médio (MSE de Treino)', fontsize=12)
plt.yscale('log') # Escala logarítmica ajuda a enxergar diferenças sutis quando o erro fica pequeno
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.legend(fontsize=11)

# Salva a figura para o seu relatório da UFES
plt.savefig('curvas_treinamento_multivariado.png', dpi=300, bbox_inches='tight')
print("\nGráfico das curvas salvo com sucesso como 'curvas_treinamento_multivariado.png'!")
plt.show()