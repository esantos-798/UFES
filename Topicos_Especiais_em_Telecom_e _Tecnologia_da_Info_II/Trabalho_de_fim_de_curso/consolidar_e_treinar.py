import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def inspecionar_e_processar_arquivo(caminho_arq, classe_falha_id):
    """
    Lê o arquivo .txt em blocos, identifica a ordem física das colunas,
    trata rigorosamente valores nulos/NaN e calcula as features e o TTF.
    """
    # Ler apenas as primeiras linhas para mapear as colunas reais
    df_header = pd.read_csv(caminho_arq, sep=r'\s+', nrows=2, engine='python')
    colunas_reais = [col.replace('"', '').strip() for col in df_header.columns]
    
    idx_time = 0 
    idx_osnr = 4 # Padrão
    
    for idx, col in enumerate(colunas_reais):
        if 'osnr' in col.lower():
            idx_osnr = idx
        elif 'time' in col.lower() or 'sample' in col.lower():
            idx_time = idx

    print(f"    Mapeamento físico -> Coluna Tempo/Sample: Índ. {idx_time} | Coluna OSNR: Índ. {idx_osnr}")
    
    chunk_size = 100000
    chunks = pd.read_csv(caminho_arq, sep=r'\s+', engine='python', header=0, chunksize=chunk_size)
    
    lista_processada = []
    
    for chunk in chunks:
        chunk.columns = [col.replace('"', '').strip() for col in chunk.columns]
        nome_col_time = chunk.columns[idx_time]
        nome_col_osnr = chunk.columns[idx_osnr]
        
        # Converte para numérico forçando NaN onde houver texto ou espaços defeituosos
        chunk[nome_col_time] = pd.to_numeric(chunk[nome_col_time], errors='coerce')
        chunk[nome_col_osnr] = pd.to_numeric(chunk[nome_col_osnr], errors='coerce')
        
        # Remove rigorosamente qualquer linha que contenha NaN ou valores não-finitos nessas duas colunas
        chunk = chunk.dropna(subset=[nome_col_time, nome_col_osnr])
        chunk = chunk[np.isfinite(chunk[nome_col_time]) & np.isfinite(chunk[nome_col_osnr])]
        
        if chunk.empty:
            continue
            
        # Agora a conversão para inteiro e float32 é 100% segura
        chunk['Sample_Time'] = chunk[nome_col_time].astype(np.int32)
        chunk['OSNR_Clean'] = chunk[nome_col_osnr].astype(np.float32)
        
        # ID de trajetória baseado na indexação sequencial original de 900 amostras
        chunk['Lightpath_ID'] = chunk.index // 900
        
        lista_processada.append(chunk[['Lightpath_ID', 'Sample_Time', 'OSNR_Clean']])
        
    df_arquivo = pd.concat(lista_processada, axis=0).reset_index(drop=True)
    
    # Engenharia de Features e Cálculo do TTF por Lightpath
    resultado_features = []
    print("    Gerando lags e derivadas por Lightpath...")
    
    for lp_id, group in df_arquivo.groupby('Lightpath_ID'):
        group = group.sort_values(by='Sample_Time').copy()
        
        max_time = group['Sample_Time'].max()
        group['TTF'] = (max_time - group['Sample_Time']).astype(np.float32)
        
        # Construção dos Lags do OSNR
        for lag in range(1, 11):
            group[f"OSNR_lag_{lag}"] = group['OSNR_Clean'].shift(lag).astype(np.float32)
            
        group["OSNR_velocity"] = group['OSNR_Clean'].diff().astype(np.float32)
        group["OSNR_acceleration"] = group["OSNR_velocity"].diff().astype(np.float32)
        group["OSNR_rolling_mean"] = group['OSNR_Clean'].rolling(window=5).mean().astype(np.float32)
        group["OSNR_rolling_std"] = group['OSNR_Clean'].rolling(window=5).std().astype(np.float32)
        
        resultado_features.append(group.dropna())
        
    return pd.concat(resultado_features, axis=0).reset_index(drop=True)

if __name__ == "__main__":
    pasta_dataset = "C:\\UFES\\Topicos_Especiais_em_Telecom_e _Tecnologia_da_Info_II\\Trabalho_de_fim_de_curso\\optical network soft failure dataset" 
    
    arq_train = os.path.join(pasta_dataset, "Lightpath_756_label_4_QoT_dataset_train_900.txt")
    arq_test = os.path.join(pasta_dataset, "Lightpath_756_label_4_QoT_dataset_test_300.txt")
    
    print("1. Processando conjunto de TREINO nativo...")
    df_train = inspecionar_e_processar_arquivo(arq_train, classe_falha_id=0)
    print(f"   -> Features de treino geradas: {df_train.shape[0]} amostras.")
    
    print("\n2. Processando conjunto de TESTE nativo...")
    df_test = inspecionar_e_processar_arquivo(arq_test, classe_falha_id=0)
    print(f"   -> Features de teste geradas: {df_test.shape[0]} amostras.")
    
    # Montar matrizes de entrada para o Random Forest
    colunas_input = ["OSNR_Clean"] + [f"OSNR_lag_{i}" for i in range(1, 11)] + \
                    ["OSNR_velocity", "OSNR_acceleration", "OSNR_rolling_mean", "OSNR_rolling_std"]
                    
    X_train = df_train[colunas_input].values
    y_train = df_train["TTF"].values
    
    X_test = df_test[colunas_input].values
    y_test = df_test["TTF"].values
    
    del df_train, df_test
    
    print("\n3. Treinando o modelo Random Forest Regressor (Baseline da Fase 1)...")
    rf = RandomForestRegressor(n_estimators=50, max_samples=0.5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("\n4. Avaliando predições no conjunto de teste...")
    predicoes = rf.predict(X_test)
    mae = mean_absolute_error(y_test, predicoes)
    
    print(f"\n[SUCESSO] MAE obtido na reprodução: {mae:.2f} passos de tempo.")