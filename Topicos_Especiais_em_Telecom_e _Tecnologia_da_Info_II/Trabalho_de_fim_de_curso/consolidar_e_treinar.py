import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def carregar_e_consolidar_otimizado(pasta_path):
    """
    Carrega o dataset em blocos (chunks) interpretando corretamente as aspas
    e limpando os nomes de colunas de forma agressiva.
    """
    arquivos = glob.glob(os.path.join(pasta_path, "*.txt"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo .txt encontrado na pasta: {pasta_path}")
        
    lista_df = []

    for id_arq, caminho_arq in enumerate(arquivos):
        nome_arq = os.path.basename(caminho_arq)
        info = nome_arq.replace(".txt", "").split("_")
        tipo_falha_nome = info[0]
        
        mapeamento_falhas = {"Normal": 0, "No": 0, "ECL": 1, "EDFA": 2, "NLI": 3}
        classe_falha = mapeamento_falhas.get(tipo_falha_nome, 0)
        
        print(f"  -> Lendo arquivo: {nome_arq} em chunks...")
        
        chunk_size = 100000
        chunks = pd.read_csv(caminho_arq, sep=r'\s+', engine='python', chunksize=chunk_size)
        
        for chunk in chunks:
            # Limpeza profunda de colunas: remove aspas, espaços extras nas pontas
            chunk.columns = [col.replace('"', '').strip() for col in chunk.columns]
            
            # Conversão dinâmica e segura de tipos para economizar RAM
            for col in chunk.columns:
                if 'time' in col.lower():
                    chunk[col] = chunk[col].astype(np.int32)
                elif 'length' in col.lower():
                    chunk[col] = chunk[col].astype(np.int16)
                elif any(x in col.lower() for x in ['current', 'power', 'osnr', 'ber']):
                    chunk[col] = chunk[col].astype(np.float32)
            
            # Adicionar identificadores compactos
            chunk["Lightpath_ID"] = np.int16(id_arq)
            chunk["Failure_type_target"] = np.int8(classe_falha)
            
            lista_df.append(chunk)
        
    print("Agrupando todos os blocos na memória...")
    df_completo = pd.concat(lista_df, axis=0).reset_index(drop=True)
    return df_completo

def calcular_ttf_e_features_leve(df):
    """
    Detecta dinamicamente as colunas de OSNR e Time stamp e calcula as features.
    """
    # Identificação dinâmica das colunas corretas baseada em buscas parciais
    col_time = [c for c in df.columns if 'time' in c.lower()][0]
    col_osnr = [c for c in df.columns if 'osnr' in c.lower()][0]
    
    print(f"Detected columns -> Time: '{col_time}' | OSNR: '{col_osnr}'")
    
    processados = []
    print("Calculando TTF e engenharia de features por Lightpath...")
    
    for lp_id, group in df.groupby("Lightpath_ID"):
        group = group.sort_values(by=col_time).copy()
        
        tempo_maximo = group[col_time].max()
        group["TTF"] = (tempo_maximo - group[col_time]).astype(np.float32)
        
        # Gerar os lags baseados no OSNR identificado
        for lag in range(1, 11):
            group[f"OSNR_lag_{lag}"] = group[col_osnr].shift(lag).astype(np.float32)
            
        group["OSNR_velocity"] = group[col_osnr].diff().astype(np.float32)
        group["OSNR_acceleration"] = group["OSNR_velocity"].diff().astype(np.float32)
        group["OSNR_rolling_mean"] = group[col_osnr].rolling(window=5).mean().astype(np.float32)
        group["OSNR_rolling_std"] = group[col_osnr].rolling(window=5).std().astype(np.float32)
        
        processados.append(group.dropna())
        
    return pd.concat(processados, axis=0).reset_index(drop=True), col_osnr

if __name__ == "__main__":
    pasta_dataset = "C:\\UFES\\Topicos_Especiais_em_Telecom_e _Tecnologia_da_Info_II\\Trabalho_de_fim_de_curso\\optical network soft failure dataset" 
    
    print("1. Consolidando de forma otimizada para a RAM...")
    df_bruto = carregar_e_consolidar_otimizado(pasta_dataset)
    print(f"Total de registros carregados: {len(df_bruto)}")
    
    print("\n2. Computando features e limpando dados temporais...")
    df_features, nome_col_osnr = calcular_ttf_e_features_leve(df_bruto)
    del df_bruto # Libera a memória imediatamente
    
    # Montar lista de inputs dinamicamente com o nome exato da coluna OSNR encontrada
    colunas_input = [nome_col_osnr] + [f"OSNR_lag_{i}" for i in range(1, 11)] + \
                    ["OSNR_velocity", "OSNR_acceleration", "OSNR_rolling_mean", "OSNR_rolling_std"]
                    
    X = df_features[colunas_input].values 
    y = df_features["TTF"].values
    grupos = df_features["Lightpath_ID"].values
    
    del df_features 
    
    print("\n3. Divisão dos conjuntos (Train/Test Split por Trajetória)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, grupos))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    
    print("\n4. Treinando o modelo Random Forest Regressor (Fase 1)...")
    rf = RandomForestRegressor(n_estimators=50, max_samples=0.5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    predicoes = rf.predict(X_test)
    mae = mean_absolute_error(y_test, predicoes)
    print(f"\n[SUCESSO] MAE obtido no seu ambiente: {mae:.2f} passos de tempo.")