import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. AJUSTE NO PRÉ-PROCESSAMENTO E JANELAMENTO
# ==========================================

def prepare_multivariate_sequences(df, window_size=10):
    """
    Processa os dados de forma multivariada e cria janelas deslizantes 3D [amostras, time_steps, features]
    garantindo que não haja cruzamento entre trajetórias diferentes.
    """
    # Identificar a coluna de agrupamento (por comprimento ou ID se houver)
    group_col = 'LP length (km)' if 'Lightpath_ID' not in df.columns else 'Lightpath_ID'
    
    # Features selecionadas para o Experimento 2 (Multivariado)
    feature_cols = ['OSNR (dB)', 'BER (dB)', 'Laser current (mA)', 'LP power (dBm)']
    
    X_seqs = []
    y_ttf = []
    y_class = []
    trajectory_groups = []
    
    for path_id, group in df.groupby(group_col):
        group = group.sort_values(by="Time stamp").copy()
        
        # Encontrar o ponto exato da falha
        failure_events = group[group["Failure type"] != 0]
        if failure_events.empty:
            continue # Pula trajetórias sem falhas registradas para o cálculo de TTF
            
        first_failure_time = failure_events["Time stamp"].min()
        
        # Calcular TTF (Time-to-Failure)
        group["TTF"] = first_failure_time - group["Time stamp"]
        
        # Filtrar para manter apenas a janela que antecede a falha (TTF >= 0)
        group_pre_failure = group[group["TTF"] >= 0].reset_index(drop=True)
        
        if len(group_pre_failure) < window_size:
            continue # Pula se a trajetória for menor que a janela de tempo necessária
            
        # Escalar as features locais da trajetória para estabilidade da LSTM
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(group_pre_failure[feature_cols])
        
        # Criar janelas deslizantes
        for i in range(len(group_pre_failure) - window_size + 1):
            window = scaled_features[i : i + window_size]
            target_ttf = group_pre_failure.loc[i + window_size - 1, "TTF"]
            target_class = group_pre_failure.loc[i + window_size - 1, "Failure type"]
            
            X_seqs.append(window)
            y_ttf.append(target_ttf)
            y_class.append(target_class)
            trajectory_groups.append(path_id)
            
    if len(X_seqs) == 0:
        raise ValueError("ERRO: O processamento gerou 0 amostras. Verifique se os 'Time stamp' e 'Failure type' estão corretos.")
        
    return np.array(X_seqs), np.array(y_ttf), np.array(y_class), np.array(trajectory_groups)

# ==========================================
# 2. PYTORCH DATASET STAGE
# ==========================================

class OpticalTelemetryDataset(Dataset):
    def __init__(self, X, y_ttf, y_class):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_ttf = torch.tensor(y_ttf, dtype=torch.float32).unsqueeze(1)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y_ttf[idx], self.y_class[idx]

# ==========================================
# 3. ARQUITETURA DA ARQUITETURA MULTI-TASK (BiLSTM)
# ==========================================

class MultiTaskBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=4):
        super(MultiTaskBiLSTM, self).__init__()
        
        # BiLSTM captura dependências temporais tanto progressivas quanto regressivas
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=2, 
                            batch_first=True, 
                            bidirectional=True)
        
        # O output da BiLSTM terá tamanho: hidden_dim * 2 (devido à bidirecionalidade)
        lstm_output_dim = hidden_dim * 2
        
        # Cabeça de Regressão: Prever o tempo até a falha (TTF)
        self.regression_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Cabeça de Classificação: Identificar o tipo de falha (0, 1, 2 ou 3)
        self.classification_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x shape: [batch, time_steps, features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Pegamos o output do último time step da sequência
        last_step_out = lstm_out[:, -1, :]
        
        ttf_pred = self.regression_head(last_step_out)
        class_pred = self.classification_head(last_step_out)
        
        return ttf_pred, class_pred

# ==========================================
# 4. PIPELINE DE EXECUÇÃO
# ==========================================

if __name__ == "__main__":
    # Dados de teste baseados na sua estrutura enviada para garantir que o bug foi sanado
    data_mock = {
        "Time stamp": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "LP length (km)": [514, 514, 514, 514, 514, 620, 620, 620, 620, 620],
        "Laser current (mA)": [40.08, 40.09, 40.12, 40.15, 40.20, 40.10, 40.09, 40.10, 40.11, 40.15],
        "LP power (dBm)": [-1.98, -1.98, -2.00, -2.02, -2.05, -2.00, -2.00, -2.01, -2.02, -2.03],
        "OSNR (dB)": [24.75, 24.75, 24.74, 24.71, 24.68, 24.74, 24.74, 24.74, 24.73, 24.71],
        "BER (dB)": [-267.8, -267.7, -267.3, -266.1, -265.0, -267.3, -267.2, -267.3, -266.9, -266.0],
        "Failure type": [0, 0, 0, 0, 2, 0, 0, 0, 0, 1] # Falhas ocorrem no timestamp 5
    }
    df = pd.DataFrame(data_mock)
    
    # Altere a linha abaixo para apontar para o seu csv completo do Mendeley:
    # df = pd.read_csv("seu_arquivo.csv")

    print("Processando dados e gerando janelas temporais multivariadas...")
    X_all, y_ttf_all, y_class_all, groups_all = prepare_multivariate_sequences(df, window_size=3)
    
    print("Dividindo o dataset por trajetórias (Garantia anti-leakage)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_ttf_all, groups_all))
    
    # Instanciando os Datasets do PyTorch
    train_dataset = OpticalTelemetryDataset(X_all[train_idx], y_ttf_all[train_idx], y_class_all[train_idx])
    test_dataset = OpticalTelemetryDataset(X_all[test_idx], y_ttf_all[test_idx], y_class_all[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    print(f"Dados prontos! Lotes de treino: {len(train_loader)} | Lotes de teste: {len(test_loader)}")
    
    # Inicializando o modelo
    model = MultiTaskBiLSTM(input_dim=4, hidden_dim=16, num_classes=4)
    
    # Definição das funções de perda para Multi-task
    criterion_regression = nn.MSELoss()          # Para o TTF
    criterion_classification = nn.CrossEntropyLoss()  # Para o Tipo de Falha
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loop rápido de treinamento demonstrativo (1 Epoch)
    model.train()
    for batch_X, batch_ttf, batch_class in train_loader:
        optimizer.zero_grad()
        
        # Forward pass obtendo as duas predições simultâneas
        pred_ttf, pred_class = model(batch_X)
        
        # Cálculo das perdas isoladas
        loss_ttf = criterion_regression(pred_ttf, batch_ttf)
        loss_class = criterion_classification(pred_class, batch_class)
        
        # Perda combinada (pode ser ponderada, ex: loss_ttf + 0.5 * loss_class)
        total_loss = loss_ttf + loss_class
        
        total_loss.backward()
        optimizer.step()
        
    print("Execução da estrutura base concluída com sucesso! Pronto para escala real.")