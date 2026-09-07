import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. TRATAMENTO, INTERPOLAÇÃO E JANELA (6h)
# ==========================================
arquivo_csv = "INMET_SE_ES_A612_VITORIA_01-01-2025_A_31-12-2025.CSV" 
df = pd.read_csv(arquivo_csv, sep=';', skiprows=8, encoding='latin-1', decimal=',')

coluna_data = [c for c in df.columns if 'Data' in c or 'DATA' in c][0]
coluna_hora = [c for c in df.columns if 'Hora' in c or 'HORA' in c][0]
df['Data_Hora'] = pd.to_datetime(df[coluna_data] + ' ' + df[coluna_hora], errors='coerce')
df.set_index('Data_Hora', inplace=True)
df = df.replace([-9999, 9999, -9999.0, 9999.0], np.nan)

coluna_vento = [c for c in df.columns if 'VENTO' in c.upper() and 'VELOCIDADE' in c.upper()][0]
df_vento = df[[coluna_vento]].copy()
df_vento[coluna_vento] = df_vento[coluna_vento].interpolate(method='linear', limit_direction='both')

df_janela = pd.DataFrame(index=df_vento.index)
for i in range(5, -1, -1):
    nome_col = f'vento_(t-{i})' if i > 0 else 'vento_atual_(t)'
    df_janela[nome_col] = df_vento[coluna_vento].shift(i)
df_janela['alvo_(t+1)'] = df_vento[coluna_vento].shift(-1)
df_janela.dropna(inplace=True)

X = df_janela.drop(columns=['alvo_(t+1)'])
y = df_janela['alvo_(t+1)']

# Divisão Cronológica
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

# Formatação 3D para LSTM no PyTorch: [Amostras, Timesteps (6), Features (1)]
X_train_3D = torch.tensor(X_train_s.reshape(-1, 6, 1), dtype=torch.float32)
y_train_3D = torch.tensor(y_train_s.reshape(-1, 1), dtype=torch.float32)
X_val_3D = torch.tensor(X_val_s.reshape(-1, 6, 1), dtype=torch.float32)
y_val_3D = torch.tensor(y_val_s.reshape(-1, 1), dtype=torch.float32)
X_test_3D = torch.tensor(X_test_s.reshape(-1, 6, 1), dtype=torch.float32)

# ==========================================
# 2. DEFINIÇÃO DA REDE BiLSTM (16 CÉLULAS POR DIREÇÃO)
# ==========================================
class VentoBiLSTM(nn.Module):
    def __init__(self):
        super(VentoBiLSTM, self).__init__()
        # bidirectional=True ativa a leitura nos dois sentidos temporais
        self.bilstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        # Como são duas direções de 16 células, a entrada da camada linear passa a ser 16 * 2 = 32
        self.linear = nn.Linear(16 * 2, 1) 
        
    def forward(self, x):
        out, _ = self.bilstm(x)
        # Na BiLSTM, concatenamos o último passo da frente com o "primeiro" de trás
        out = self.linear(out[:, -1, :])
        return out

modelo_lstm = VentoBiLSTM() # Instancia a nova rede bidirecional
criterio = nn.MSELoss()

# Mantemos a mesma configuração de L1 e L2 anterior para comparação justa
otimizador = optim.Adam(modelo_lstm.parameters(), lr=0.005, weight_decay=1e-4)
lambda_l1 = 1e-5

# ==========================================
# 3. LOOPS DE TREINAMENTO E VALIDAÇÃO
# ==========================================
historico_treino, historico_val = [], []
epocas = 60

print("Treinando a BiLSTM Bidirecional...")
for epoca in range(epocas):
    modelo_lstm.train()
    otimizador.zero_grad()
    saida = modelo_lstm(X_train_3D)
    
    loss_base = criterio(saida, y_train_3D)
    
    # Regularização L1
    l1_regularization = 0
    for param in modelo_lstm.parameters():
        l1_regularization += torch.sum(torch.abs(param))
        
    loss_treino = loss_base + lambda_l1 * l1_regularization
    loss_treino.backward()
    otimizador.step()
    
    # Validação
    modelo_lstm.eval()
    with torch.no_grad():
        saida_val = modelo_lstm(X_val_3D)
        loss_val = criterio(saida_val, y_val_3D)
        
    historico_treino.append(loss_base.item())
    historico_val.append(loss_val.item())
    
    if (epoca + 1) % 10 == 0:
        print(f"Época [{epoca+1}/{epocas}] | Erro Treino (Base): {loss_base.item():.5f} | Erro Val: {loss_val.item():.5f}")

# ==========================================
# 4. PREVISÃO E MÉTRICAS DE ERRO
# ==========================================
modelo_lstm.eval()
with torch.no_grad():
    preds_test_s = modelo_lstm(X_test_3D).numpy()

preds_test = scaler_y.inverse_transform(preds_test_s).flatten()

mae = mean_absolute_error(y_test, preds_test)
rmse = np.sqrt(mean_squared_error(y_test, preds_test))

print("\n==========================================")
print("     RESULTADO DA SUA LSTM (16 CÉLULAS)    ")
print("==========================================")
print(f"MAE  (Teste): {mae:.4f} m/s")
print(f"RMSE (Teste): {rmse:.4f} m/s\n")

# ==========================================
# 5. GERANDO A CURVA DE TREINAMENTO
# ==========================================

plt.figure(figsize=(10, 5))
plt.plot(historico_treino, label='Erro de Treinamento (Loss)', color='teal', linewidth=2)
plt.plot(historico_val, label='Erro de Validação', color='orange', linestyle='--', linewidth=2)
plt.title('Curva de Treinamento da LSTM (16 Células - Janela 6h)', fontsize=13, fontweight='bold')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.savefig('curva_treinamento_lstm16.png', dpi=300)
print("Gráfico da curva salvo como 'curva_treinamento_lstm16.png'.")
plt.show()

