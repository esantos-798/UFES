import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def carregar_dados_multivariados(caminho_arq):
    df_header = pd.read_csv(caminho_arq, sep=r'\s+', nrows=2, engine='python')
    colunas_reais = [col.replace('"', '').strip() for col in df_header.columns]
    
    idx_time, idx_current, idx_power, idx_osnr, idx_ber = 0, 2, 3, 4, 5
    for idx, col in enumerate(colunas_reais):
        c_low = col.lower()
        if 'time' in c_low or 'sample' in c_low: idx_time = idx
        elif 'current' in c_low: idx_current = idx
        elif 'power' in c_low: idx_power = idx
        elif 'osnr' in c_low: idx_osnr = idx
        elif 'ber' in c_low: idx_ber = idx

    chunk_size = 100000
    chunks = pd.read_csv(caminho_arq, sep=r'\s+', engine='python', header=0, chunksize=chunk_size)
    
    lista_df = []
    for chunk in chunks:
        chunk.columns = [col.replace('"', '').strip() for col in chunk.columns]
        c_time, c_current, c_power, c_osnr, c_ber = chunk.columns[idx_time], chunk.columns[idx_current], chunk.columns[idx_power], chunk.columns[idx_osnr], chunk.columns[idx_ber]
        
        for col in [c_time, c_current, c_power, c_osnr, c_ber]:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
        chunk = chunk.dropna(subset=[c_time, c_current, c_power, c_osnr, c_ber])
        if chunk.empty: continue
            
        df_bloco = pd.DataFrame({
            'Lightpath_ID': chunk.index // 900,
            'Time': chunk[c_time].astype(np.int32),
            'Current': chunk[c_current].astype(np.float32),
            'Power': chunk[c_power].astype(np.float32),
            'OSNR': chunk[c_osnr].astype(np.float32),
            'BER': chunk[c_ber].astype(np.float32)
        })
        lista_df.append(df_bloco)
        
    return pd.concat(lista_df, axis=0).reset_index(drop=True)

def construir_features_sequenciais(df, window_size=10):
    X_list, y_list = [], []
    features_cols = ['OSNR', 'BER', 'Power', 'Current']
    
    print(f"    Gerando hist�rico temporal profundo ({window_size} lags) para vari�veis multivariadas...")
    for lp_id, group in df.groupby('Lightpath_ID'):
        group = group.sort_values(by='Time').copy()
        
        max_time = group['Time'].max()
        group['TTF'] = max_time - group['Time']
        
        for col in features_cols:
            for lag in range(1, window_size + 1):
                group[f"{col}_lag_{lag}"] = group[col].shift(lag)
                
        group = group.dropna()
        colunas_input = features_cols + [c for c in group.columns if '_lag_' in c]
        X_list.append(group[colunas_input].values)
        y_list.append(group['TTF'].values)
        
    return np.vstack(X_list), np.concatenate(y_list)

if __name__ == "__main__":
    pasta_dataset = "C:\\UFES\\Topicos_Especiais_em_Telecom_e _Tecnologia_da_Info_II\\Trabalho_de_fim_de_curso\\optical network soft failure dataset" 
    arq_train = os.path.join(pasta_dataset, "Lightpath_756_label_4_QoT_dataset_train_900.txt")
    arq_test = os.path.join(pasta_dataset, "Lightpath_756_label_4_QoT_dataset_test_300.txt")
    
    print("1. Carregando dados brutos...")
    df_train_raw = carregar_dados_multivariados(arq_train)
    df_test_raw = carregar_dados_multivariados(arq_test)
    
    print("\n2. Criando janelas temporais de 10 passos...")
    X_train, y_train = construir_features_sequenciais(df_train_raw, window_size=10)
    X_test, y_test = construir_features_sequenciais(df_test_raw, window_size=10)
    del df_train_raw, df_test_raw
    
    print("\n3. Normalizando os dados...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"   Shape Treino: {X_train.shape} | Shape Teste: {X_test.shape}")
    
    print("\n4. Treinando Rede Neural Otimizada (Fase 2)...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        batch_size=4096,
        learning_rate_init=0.005,
        max_iter=40,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42,
        verbose=True
    )
    mlp.fit(X_train, y_train)
    
    print("\n5. Avaliando resultados...")
    predicoes = mlp.predict(X_test)
    mae_mlp = mean_absolute_error(y_test, predicoes)
    print(f"\n[SUCESSO] MAE final obtido: {mae_mlp:.2f} passos de tempo.")
