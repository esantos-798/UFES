import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def pre_process_and_compute_ttf(df):
    """
    Identifica cada trajetória, calcula o Time-to-Failure (TTF) baseado no momento
    em que a falha ocorre, e filtra os dados para a tarefa de regressão (janela pré-falha).
    """
    processed_paths = []
    
    # Se o dataset não tiver 'Lightpath_ID', agrupamos por 'LP length (km)' ou outra ID que diferencie as simulações
    # Caso você tenha uma coluna 'Lightpath_ID' no arquivo completo, mude aqui.
    group_col = 'LP length (km)' if 'Lightpath_ID' not in df.columns else 'Lightpath_ID'
    
    for path_id, group in df.groupby(group_col):
        group = group.sort_values(by="Time stamp").copy()
        
        # Identificar onde a falha começa (Failure type muda de 0 para 1, 2 ou 3)
        failure_indices = group[group["Failure type"] != 0]["Time stamp"]
        
        if not failure_indices.empty:
            first_failure_timestamp = failure_indices.min()
            # TTF em unidades de steps temporais (ou multiplicado pelo intervalo real em segundos)
            group["TTF"] = first_failure_timestamp - group["Time stamp"]
            
            # O artigo costuma prever o TTF apenas nas janelas que antecedem a falha (onde TTF >= 0)
            group = group[group["TTF"] >= 0]
            processed_paths.append(group)
        else:
            # Se for uma trajetória totalmente saudável, dependendo do artigo, ela pode ser descartada
            # na regressão de TTF ou receber um valor teto. Vamos manter apenas trajetórias com falha para o TTF.
            pass
            
    if not processed_paths:
        raise ValueError("Nenhuma falha encontrada no subset de dados para calcular o TTF.")
        
    return pd.concat(processed_paths, axis=0).reset_index(drop=True)

def generate_article_features(df):
    """
    Gera exatamente as 15 features baseadas puramente no OSNR (dB)
    """
    features_list = []
    group_col = 'LP length (km)' if 'Lightpath_ID' not in df.columns else 'Lightpath_ID'
    
    for path_id, group in df.groupby(group_col):
        group = group.copy()
        
        # 1. OSNR Atual: já é a coluna 'OSNR (dB)'
        
        # 2. Lags de OSNR (1 a 10 valores anteriores)
        for lag in range(1, 11):
            group[f'OSNR_lag_{lag}'] = group['OSNR (dB)'].shift(lag)
            
        # 3. Velocidade de variação (Primeira derivada)
        group['OSNR_velocity'] = group['OSNR (dB)'].diff()
        
        # 4. Aceleração (Segunda derivada)
        group['OSNR_acceleration'] = group['OSNR_velocity'].diff()
        
        # 5. Média móvel (Janela de 5 lags anteriores)
        group['OSNR_rolling_mean'] = group['OSNR (dB)'].rolling(window=5).mean()
        
        # 6. Desvio padrão móvel (Janela de 5 lags anteriores)
        group['OSNR_rolling_std'] = group['OSNR (dB)'].rolling(window=5).std()
        
        features_list.append(group)
        
    df_features = pd.concat(features_list, axis=0)
    # Remove linhas que ficaram com NaN devido ao deslocamento temporal (lags)
    return df_features.dropna().reset_index(drop=True)

# --- PIPELINE DE EXECUÇÃO ---

# 1. Carregar seus dados (Substitua pelo seu arquivo real)
# df = pd.read_csv("seu_dataset_mendeley.csv")

# Exemplo de criação de DataFrame estruturado com os dados que você enviou para testar a execução:
data_exemplo = {
    "Time stamp": [1, 2, 3, 1, 2, 3],
    "LP length (km)": [514, 514, 514, 620, 620, 620], # Simulando duas trajetórias diferentes pelos comprimentos
    "Laser current (mA)": [40.086, 40.095, 40.125, 40.109, 40.096, 40.108],
    "LP power (dBm)": [-1.985, -1.989, -2.002, -2.001, -2.005, -2.001],
    "OSNR (dB)": [24.754, 24.752, 24.745, 24.746, 24.744, 24.746],
    "BER (dB)": [-267.82, -267.71, -267.35, -267.38, -267.24, -267.38],
    "Failure type": [0, 0, 2, 0, 0, 1] # Simulando que a falha ocorre no timestamp 3
}
df = pd.DataFrame(data_exemplo)

print("Calculando a variável alvo (TTF)...")
df_ttf = pre_process_and_compute_ttf(df)

print("Gerando as 15 features baseadas no artigo...")
df_features = generate_article_features(df_ttf)

# Definir a lista exata de colunas que o modelo vai usar
colunas_features = ['OSNR (dB)'] + [f'OSNR_lag_{i}' for i in range(1, 11)] + \
                   ['OSNR_velocity', 'OSNR_acceleration', 'OSNR_rolling_mean', 'OSNR_rolling_std']

X = df_features[colunas_features]
y = df_features['TTF']
# Evitar data leakage agrupando estritamente pelo identificador da trajetória
grupos = df_features['LP length (km)'] 

print("Separando os dados por Trajetória (Train/Test Split)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, grupos))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# Inicializar e treinar o Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Avaliação preliminar
predicoes = rf.predict(X_test)
mae = mean_absolute_error(y_test, predicoes)
print(f"MAE na reprodução do Baseline: {mae:.4f}")