# src/data_processing.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def load_and_engineer_features(filepath):
    df_raw = pd.read_csv(filepath)
    
    # Se o arquivo já tiver a coluna 'OutputPower_Ampli1', ele pula o pivot_table
    if 'OutputPower_Ampli1' in df_raw.columns:
        df_pivot = df_raw
    else:
        # Bloco original do seu código para o dataset do artigo
        df_pivot = df_raw.pivot_table(
            index='Timestamp',
            columns='ID',
            values=['BER', 'OSNR', 'InputPower', 'OutputPower']
        )
        df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
        df_pivot = df_pivot.reset_index()
    
    # Daqui para baixo, o seu código original continua igual:
    df_pivot['Link_amp1_amp2'] = df_pivot['OutputPower_Ampli1'] - df_pivot['InputPower_Ampli2']
    df_pivot['Link_amp2_amp3'] = df_pivot['OutputPower_Ampli2'] - df_pivot['InputPower_Ampli3']
    df_pivot['Link_amp3_amp4'] = df_pivot['OutputPower_Ampli3'] - df_pivot['InputPower_Ampli4']
    
    return df_pivot

def prepare_datasets(df, window_size, batch_size):
    """Divide os dados (70/10/20), normaliza e cria os DataLoaders."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    
    # Remove a coluna de timestamp e colunas de texto se sobrarem
    features = [col for col in df.columns if col not in ['Timestamp', 'Failure']]
    data_mat = df[features].values
    
    train_data = data_mat[:train_end]
    val_data = data_mat[train_end:val_end]
    test_data = data_mat[val_end:]
    
    # Ajuste do Scaler apenas no Treino
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    def create_windows(data):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)
    
    X_train, y_train = create_windows(train_scaled)
    X_val, y_val = create_windows(val_scaled)
    X_test, y_test = create_windows(test_scaled)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, (X_test, y_test), scaler, features