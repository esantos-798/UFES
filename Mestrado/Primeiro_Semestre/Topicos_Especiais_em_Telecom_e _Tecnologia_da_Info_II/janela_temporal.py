import pandas as pd
import numpy as np

def criar_janela_temporal(dataframe, tamanho_janela=1, horizonte=1):
    """
    Cria uma janela temporal para a previsão da velocidade do vento.
    
    Parâmetros:
    - dataframe: DataFrame original do INMET.
    - tamanho_janela: Quantidade de horas do passado (lags) para usar como entrada.
    - horizonte: Quantas horas no futuro queremos prever (ex: 1 para a próxima hora).
    
    Retorna:
    - X: DataFrame com as variáveis de entrada.
    - y: Series com a variável alvo.
    """
    df_temp = dataframe.copy()
    
    # 1. Identifica a coluna de vento dinamicamente
    coluna_vento = [c for c in df_temp.columns if 'VENTO' in c.upper() and 'VELOCIDADE' in c.upper()][0]
    
    # 2. Cria uma estrutura apenas para o vento e remove nulos
    df_vento = df_temp[[coluna_vento]].dropna().copy()
    
    colunas_X = []
    
    # 3. Cria os lags do passado (t, t-1, t-2...) com base no tamanho_janela
    for i in range(tamanho_janela - 1, -1, -1):
        nome_col = f'vento_(t-{i})' if i > 0 else 'vento_atual_(t)'
        df_vento[nome_col] = df_vento[coluna_vento].shift(i)
        colunas_X.append(nome_col)
        
    # 4. Cria a coluna alvo do futuro (t + horizonte)
    nome_alvo = f'vento_futuro_(t+{horizonte})'
    df_vento[nome_alvo] = df_vento[coluna_vento].shift(-horizonte)
    
    # 5. Remove as linhas que ficaram com NaN devido aos shifts
    df_vento.dropna(inplace=True)
    
    # 6. Separa X e y
    X = df_vento[colunas_X]
    y = df_vento[nome_alvo]
    
    return X, y