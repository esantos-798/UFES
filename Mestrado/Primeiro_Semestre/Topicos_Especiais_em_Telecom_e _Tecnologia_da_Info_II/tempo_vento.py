import pandas as pd
import numpy as np
from janela_temporal import criar_janela_temporal

# 1. Carrega o dataset e faz o tratamento básico (reaproveitando o código anterior)
df = pd.read_csv("INMET_SE_ES_A612_VITORIA_01-01-2025_A_31-12-2025.CSV", sep=';', skiprows=8, encoding='latin-1', decimal=',')
coluna_data = [c for c in df.columns if 'Data' in c or 'DATA' in c][0]
coluna_hora = [c for c in df.columns if 'Hora' in c or 'HORA' in c][0]

df['Data_Hora'] = pd.to_datetime(df[coluna_data] + ' ' + df[coluna_hora], errors='coerce')
df.set_index('Data_Hora', inplace=True)
df = df.replace([-9999, 9999, -9999.0, 9999.0], np.nan)

# Identifica e extrai a coluna do vento
coluna_vento = [c for c in df.columns if 'VENTO' in c.upper() and 'VELOCIDADE' in c.upper()][0]
df_vento = df[[coluna_vento]].copy()

# Remove linhas com valores nulos no vento para não quebrar o modelo
df_vento.dropna(inplace=True)

# --- CRIANDO A JANELA TEMPORAL ---

# X (Feature): O vento na hora atual (t)
df_vento['vento_atual_(t)'] = df_vento[coluna_vento]

# Y (Target): O vento na próxima hora (t+1) -> Deslocamos a coluna para trás usando .shift(-1)
df_vento['vento_proxima_hora_(t+1)'] = df_vento[coluna_vento].shift(-1)

# Como o último registro não terá uma "próxima hora", ele vira NaN. Vamos removê-lo.
df_vento.dropna(inplace=True)

# Exibe o resultado da janela
print("Dataset estruturado para Previsão (Janela de 1h):")
print(df_vento[['vento_atual_(t)', 'vento_proxima_hora_(t+1)']].head())

# 1. Separando as variáveis X (entrada/passado) e y (alvo/futuro)
X = df_vento[['vento_atual_(t)']]
y = df_vento['vento_proxima_hora_(t+1)']

# 2. Concatenando lado a lado para verificação
verificacao = pd.concat([X, y], axis=1)

# Configuração opcional do Pandas para garantir que apareçam todas as linhas no print
pd.set_option('display.max_rows', 20)

print("=== VERIFICAÇÃO DO DELAY DA JANELA TEMPORAL ===")
print(verificacao.head(15))


# --- CHAMANDO A FUNÇÃO ---

# Janela de 1 hora: Usa apenas a hora atual (t) para prever a próxima (t+1)
X, y = criar_janela_temporal(df, tamanho_janela=1, horizonte=1)

# Verificação lado a lado
print("=== X (Features) ===")
print(X.head())
print("\n=== y (Alvo) ===")
print(y.head())


# 1. Descobre as datas de corte baseadas nos índices ordenados por tempo
datas = X.index

# Define os índices inteiros de corte (70% e 85%)
idx_70 = int(len(datas) * 0.70)
idx_85 = int(len(datas) * 0.85)

# Pega a data exata onde terminam o treino e a validação
data_fim_treino = datas[idx_70]
data_fim_val = datas[idx_85]

# 2. Divisão cronológica usando as fatias de data (Sem qualquer função randômica)
X_train = X.loc[X.index < data_fim_treino]
y_train = y.loc[y.index < data_fim_treino]

X_val = X.loc[(X.index >= data_fim_treino) & (X.index < data_fim_val)]
y_val = y.loc[(y.index >= data_fim_treino) & (y.index < data_fim_val)]

X_test = X.loc[X.index >= data_fim_val]
y_test = y.loc[y.index >= data_fim_val]

# --- IMPRESSÃO DE SEGURANÇA ---
print("=== VERIFICAÇÃO DE SEGURANÇA CRONOLÓGICA ===")
print(f"Treino:    {X_train.index.min()} até {X_train.index.max()} | Total: {len(X_train)}")
print(f"Validação: {X_val.index.min()} até {X_val.index.max()} | Total: {len(X_val)}")
print(f"Teste:     {X_test.index.min()} até {X_test.index.max()} | Total: {len(X_test)}")


# 1. Descobre as datas de corte baseadas nos índices ordenados por tempo
datas = X.index

# Define os índices inteiros de corte (70% e 85%)
idx_70 = int(len(datas) * 0.70)
idx_85 = int(len(datas) * 0.85)

# Pega a data exata onde terminam o treino e a validação
data_fim_treino = datas[idx_70]
data_fim_val = datas[idx_85]

# 2. Divisão cronológica usando as fatias de data (Sem qualquer função randômica)
X_train = X.loc[X.index < data_fim_treino]
y_train = y.loc[y.index < data_fim_treino]

X_val = X.loc[(X.index >= data_fim_treino) & (X.index < data_fim_val)]
y_val = y.loc[(y.index >= data_fim_treino) & (y.index < data_fim_val)]

X_test = X.loc[X.index >= data_fim_val]
y_test = y.loc[y.index >= data_fim_val]

# --- IMPRESSÃO DE SEGURANÇA ---
print("=== VERIFICAÇÃO DE SEGURANÇA CRONOLÓGICA ===")
print(f"Treino:    {X_train.index.min()} até {X_train.index.max()} | Total: {len(X_train)}")
print(f"Validação: {X_val.index.min()} até {X_val.index.max()} | Total: {len(X_val)}")
print(f"Teste:     {X_test.index.min()} até {X_test.index.max()} | Total: {len(X_test)}")

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Instanciar o modelo XGBoost para Regressão
# Usamos early_stopping_rounds com o dataset de validação para evitar overfitting
modelo_xgb = XGBRegressor(
    n_estimators=1000,     # Número máximo de árvores
    learning_rate=0.05,    # Taxa de aprendizado
    max_depth=5,           # Profundidade máxima das árvores
    random_state=42        # Semente para reprodutibilidade dos pesos internos
)

# 2. Treinar o modelo
# Passamos o conjunto de validação para monitorar o erro e parar o treino no momento certo
modelo_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False  # Mude para True se quiser ver o erro caindo a cada árvore
)

# 3. Fazer previsões nos três conjuntos (Entrada e Saída)
preds_train = modelo_xgb.predict(X_train)
preds_val = modelo_xgb.predict(X_val)
preds_test = modelo_xgb.predict(X_test)

# 4. Função para calcular e exibir os erros de forma organizada
def calcular_metricas(y_real, y_pred, nome_conjunto):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    print(f"--- Métricas de Erro: {nome_conjunto} ---")
    print(f"MAE  (Erro Médio Absoluto): {mae:.4f} m/s")
    print(f"RMSE (Raiz do Erro Quadrático): {rmse:.4f} m/s\n")

# 5. Imprimir os resultados para avaliar o desempenho
calcular_metricas(y_train, preds_train, "Treinamento")
calcular_metricas(y_val, preds_val, "Validação")
calcular_metricas(y_test, preds_test, "Teste")


