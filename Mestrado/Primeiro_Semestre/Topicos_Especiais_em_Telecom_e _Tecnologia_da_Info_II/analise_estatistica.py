import pandas as pd

# 1. Carregar os datasets
try:
    df_train = pd.read_csv('dataset_treinamento_split.csv')
    df_test = pd.read_csv('dataset_teste_split.csv')
    
    # Juntar ambos para fazer a análise do dataset completo
    df_completo = pd.concat([df_train, df_test], axis=0)
    print("Métricas Estatísticas do Dataset Completo (Normalizado):")
except FileNotFoundError:
    # Caso queira rodar direto no arquivo original sem split
    df_completo = pd.read_csv('dataset_normalizado.csv') # ou o nome do seu arquivo principal
    print("Métricas Estatísticas do Dataset:")

# 2. Selecionar apenas as colunas relevantes
colunas = ['Var_A', 'Var_B', 'Var_C', 'Var_D', 'Alvo_Y']
df_analise = df_completo[colunas]

# 3. Calcular a estatística descritiva
# .describe() traz: count, mean (média), std (desvio padrão), min, 25%, 50% (mediana), 75%, max
estatisticas = df_analise.describe().loc[['mean', 'std', 'min', '50%', 'max']]

# Renomear o índice para português ficar elegante no relatório
estatisticas.index = ['Média', 'Desvio Padrão', 'Mínimo', 'Mediana (50%)', 'Máximo']

# 4. Exibir o resultado formatado com 4 casas decimais
print("\n", estatisticas.round(4).to_string())

# 5. Opcional: Salvar em um arquivo de texto para colocar no relatório
estatisticas.round(4).to_csv('resumo_estatistico.csv', sep=';')
print("\nResumo salvo com sucesso em 'resumo_estatistico.csv'!")