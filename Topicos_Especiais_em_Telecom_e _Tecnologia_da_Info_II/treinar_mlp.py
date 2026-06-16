import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def carregar_dados_multivariados(caminho_arq):
    """
    Lê o arquivo bruto em blocos de forma otimizada para a RAM
    e extrai as colunas físicas de telemetria óptica.
    """
    chunk_size = 100000
    chunks = pd.read_csv(caminho_arq, sep=r'\s+', engine='python', header=0, chunksize=chunk_size)
    
    lista_df = []
    for chunk in chunks:
        chunk.columns = [col.replace('"', '').strip() for col in chunk.columns]
        
        # Identificação dinâmica das colunas por sub-strings
        col_time = [c for c in chunk.columns if 'time' in c.lower() or 'sample' in c.lower()][0]
        col_osnr = [c for c in chunk.columns if 'osnr' in c.lower()][0]
        col_ber = [c for c in chunk.columns if 'ber' in c.lower()][0]
        col_power = [c for c in chunk.columns if 'power' in c.lower()][0]
        col_current = [c for c in chunk.columns if 'current' in c.lower()][0]
        
        # Coergir falhas de leitura para NaN e limpar
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

def construir_features_sequenciais(df, window_size=5):
    """
    Achata o histórico temporal (lags) de todas as variáveis físicas
    para criar um preditor multivariado robusto sem depender de tensores 3D.
    """
    X_list, y_list = [], []
    features_cols = ['OSNR', 'BER', 'Power', 'Current']
    
    print("    Gerando histórico temporal (lags) para variáveis multivariadas...")
    for lp_id, group in df.groupby('Lightpath_ID'):
        group = group.sort_values(by='Time').copy()
        
        max_time = group['Time'].max()
        group['TTF'] = max_time - group['Time']
        
        # Criar os atrasos (lags) para cada uma das 4 variáveis
        for col in features_cols:
            for lag in range(1, window_size + 1):
                group[f"{col}_lag_{lag}"] = group[col].shift(lag)
                
        group = group.dropna()
        
        # Montar a matriz combinada
        colunas_input = features_cols + [c for c in group.columns if '_lag_' in c]
        X_list.append(group[colunas_input].values)
        y_list.append(group['TTF'].values)
        
    return np.vstack(X_list), np.concatenate(y_list)

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    pasta_dataset = "C:\\UFES\\Topicos_Especiais_em_Telecom_e _Tecnologia_da_Info_II\\Trabalho_de_fim_de_curso\\optical network soft failure dataset" 
    
    arq_train = os.path.join(pasta_dataset, "Lightpath_756_label_4_QoT_dataset_train_900.txt")
    arq_test = os.path.join(pasta_dataset, "Lightpath_756_label_4_QoT_dataset_test_300.txt")
    
    print("1. Carregando dados brutos multivariados (OSNR, BER, Power, Current)...")
    df_train_raw = carregar_dados_multivariados(arq_train)
    df_test_raw = carregar_dados_multivariados(arq_test)
    
    print("\n2. Criando janelas temporais acopladas...")
    X_train, y_train = construir_features_sequenciais(df_train_raw, window_size=5)
    X_test, y_test = construir_features_sequenciais(df_test_raw, window_size=5)
    
    del df_train_raw, df_test_raw # Libera RAM imediatamente
    
    print("\n3. Normalizando os dados para a Rede Neural (Essencial para convergir)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"   Shape Treino: {X_train.shape} | Shape Teste: {X_test.shape}")
    
    print("\n4. Inicializando e Treinando a Rede Neural Multi-Layer Perceptron (Fase 2)...")
    # Usando arquitetura profunda [64, 32] com early stopping para não travar a máquina
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        batch_size=2048,
        learning_rate_init=0.01,
        max_iter=15,
        early_stopping=True,
        random_state=42,
        verbose=True
    )
    
    mlp.fit(X_train, y_train)
    
    print("\n5. Avaliando a Rede Neural Multivariada no conjunto de teste...")
    predicoes = mlp.predict(X_test)
    mae_mlp = mean_absolute_error(y_test, predicoes)
    
    print(f"\n[SUCESSO] MAE obtido com a Rede Neural Multivariada: {mae_mlp:.2f} passos de tempo.")