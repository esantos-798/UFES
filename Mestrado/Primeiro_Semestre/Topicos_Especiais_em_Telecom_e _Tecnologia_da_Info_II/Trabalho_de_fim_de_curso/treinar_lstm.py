import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Configurar dispositivo (GPU se disponível, senão CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando o dispositivo: {device}")

def carregar_dados_lstm(caminho_arq):
    """
    Lê o arquivo bruto em blocos e extrai as colunas multivariadas para a LSTM.
    """
    chunk_size = 100000
    chunks = pd.read_csv(caminho_arq, sep=r'\s+', engine='python', header=0, chunksize=chunk_size)
    
    lista_df = []
    for chunk in chunks:
        chunk.columns = [col.replace('"', '').strip() for col in chunk.columns]
        
        # Identificação das colunas multivariadas via busca parcial por nome
        col_time = [c for c in chunk.columns if 'time' in c.lower() or 'sample' in c.lower()][0]
        col_osnr = [c for c in chunk.columns if 'osnr' in c.lower()][0]
        col_ber = [c for c in chunk.columns if 'ber' in c.lower()][0]
        col_power = [c for c in chunk.columns if 'power' in c.lower()][0]
        col_current = [c for c in chunk.columns if 'current' in c.lower()][0]
        
        # Converter para float32 numérico limpando falhas
        for col in [col_time, col_osnr, col_ber, col_power, col_current]:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
        chunk = chunk.dropna(subset=[col_time, col_osnr, col_ber, col_power, col_current])
        
        df_bloco = pd.DataFrame({
            'Lightpath_ID': chunk.index // 900,
            'Time': chunk[col_time].astype(np.int32),
            'OSNR': chunk[col_osnr].astype(np.float32),
            'BER': chunk[col_ber].astype(np.float32),
            'Power': chunk[col_power].astype(np.float32),
            'Current': chunk[col_current].astype(np.float32)
        })
        lista_df.append(df_bloco)
        
    return pd.concat(lista_df, axis=0).reset_index(drop=True)

def criar_janelas_3d(df, window_size=10):
    """
    Transforma o DataFrame em matrizes 3D prontas para a LSTM [Amostras, Timesteps, Features]
    Impedindo vazamento temporal entre diferentes trajetórias (Lightpaths).
    """
    X_list, y_list = [], []
    features_cols = ['OSNR', 'BER', 'Power', 'Current']
    
    for lp_id, group in df.groupby('Lightpath_ID'):
        group = group.sort_values(by='Time')
        
        # Calcular a variável alvo (TTF)
        max_time = group['Time'].max()
        group['TTF'] = max_time - group['Time']
        
        arr_features = group[features_cols].values
        arr_ttf = group['TTF'].values
        
        # Criar fatias sequenciais (sliding window)
        for i in range(len(group) - window_size):
            X_list.append(arr_features[i : i + window_size])
            y_list.append(arr_ttf[i + window_size])
            
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# --- Definição da Arquitetura de Rede LSTM ---
class RegressorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(RegressorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: [Batch, Timesteps, Features]
        out, _ = self.lstm(x)
        # Pegar apenas a saída do último timestep da janela
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

# ==========================================
# PIPELINE PRINCIPAL DE EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    pasta_dataset = "C:\\UFES\\Topicos_Especiais_em_Telecom_e _Tecnologia_da_Info_II\\Trabalho_de_fim_de_curso\\optical network soft failure dataset" 
    
    arq_train = os.path.join(pasta_dataset, "Lightpath_756_label_4_QoT_dataset_train_900.txt")
    arq_test = os.path.join(pasta_dataset, "Lightpath_756_label_4_QoT_dataset_test_300.txt")
    
    print("1. Carregando dados brutos multivariados...")
    df_train_raw = carregar_dados_lstm(arq_train)
    df_test_raw = carregar_dados_lstm(arq_test)
    
    print("2. Normalizando os dados físicos (StandardScaler)...")
    scaler = StandardScaler()
    features_cols = ['OSNR', 'BER', 'Power', 'Current']
    
    df_train_raw[features_cols] = scaler.fit_transform(df_train_raw[features_cols])
    df_test_raw[features_cols] = scaler.transform(df_test_raw[features_cols])
    
    print("3. Construindo as matrizes tridimensionais (Janela Deslizante)...")
    X_train, y_train = criar_janelas_3d(df_train_raw, window_size=10)
    X_test, y_test = criar_janelas_3d(df_test_raw, window_size=10)
    
    del df_train_raw, df_test_raw # Liberar espaço
    print(f"   Shape Treino: {X_train.shape} | Shape Teste: {X_test.shape}")
    
    # Criar DataLoaders do PyTorch
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
    
    print("4. Inicializando modelo LSTM no PyTorch...")
    modelo = RegressorLSTM(input_size=4, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=0.005)
    
    print("5. Iniciando Treinamento Rápido (3 Epocas de validação)...")
    modelo.train()
    for epoch in range(1, 4):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = modelo(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
        
        print(f"   Época {epoch}/3 finalizada.")
        
    print("\n6. Avaliando a LSTM Multivariada no conjunto de teste...")
    modelo.eval()
    todas_predicoes = []
    comprimento_real = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            pred = modelo(batch_X)
            todas_predicoes.extend(pred.cpu().numpy())
            comprimento_real.extend(batch_y.numpy())
            
    mae_lstm = mean_absolute_error(comprimento_real, todas_predicoes)
    print(f"\n[SUCESSO] MAE obtido com a LSTM Multivariada: {mae_lstm:.2f} passos de tempo.")