import pandas as pd

# Caminho para o arquivo CSV extraído do ZIP do INMET
caminho_arquivo = "INMET_SE_ES_A612_VITORIA_01-01-2025_A_31-12-2025.CSV"

# 1. Lendo os metadados da estação (opcional, apenas se quiser saber a latitude/longitude)
with open(caminho_arquivo, 'r', encoding='latin-1') as f:
    metadados = [next(f).strip() for _ in range(8)]
print("Informações da Estação:", metadados)

# 2. Carregando os dados meteorológicos reais
# skiprows=8 pula o cabeçalho. sep=';' define o separador correto.
df = pd.read_csv(caminho_arquivo, sep=';', skiprows=8, encoding='latin-1')

# Visualizar as primeiras linhas
print(df.head())