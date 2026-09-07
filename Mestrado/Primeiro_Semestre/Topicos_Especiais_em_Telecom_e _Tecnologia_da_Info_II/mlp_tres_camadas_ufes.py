import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

print("=== INICIALIZANDO MLP DE 3 CAMADAS (4 VARIÁVEIS) ===")

# 1. Carregar os datasets das 4 variáveis de decisão
df_train = pd.read_csv('dataset_treinamento_split.csv')
df_test = pd.read_csv('dataset_teste_split.csv')

colunas_x = ['Var_A', 'Var_B', 'Var_C', 'Var_D']
X_train = torch.tensor(df_train[colunas_x].values, dtype=torch.float32)
y_train = torch.tensor(df_train['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(df_test[colunas_x].values, dtype=torch.float32)
y_test = torch.tensor(df_test['Alvo_Y'].values, dtype=torch.float32).reshape(-1, 1)

# 2. Arquitetura Clássica de 3 Camadas
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # Entrada (4 neurônios) -> Camada Oculta (16 neurônios)
        self.camada_escondida = nn.Linear(input_dim, hidden_dim)
        self.ativacao = nn.Tanh() # Tanh funciona muito bem para sinais oscilatórios
        # Camada Oculta (16 neurônios) -> Saída (1 neurônio)
        self.camada_saida = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.camada_escondida(x)
        out = self.ativacao(out)
        out = self.camada_saida(out)
        return out

# Instanciando o modelo
modelo_mlp = MLP(input_dim=4, hidden_dim=16, output_dim=1)

# 3. Otimização e Critério de Erro
criterion = nn.MSELoss()
optimizer = optim.Adam(modelo_mlp.parameters(), lr=0.01)

# 4. Loop de Treinamento (250 Épocas para convergência suave)
print("-> Treinando a rede neural tradicional...")
modelo_mlp.train()
for epoch in range(250):
    optimizer.zero_grad()
    outputs = modelo_mlp(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# 5. Avaliação do Modelo no Dataset de Teste
modelo_mlp.eval()
with torch.no_grad():
    y_pred_torch = modelo_mlp(X_test)
    y_pred_mlp = y_pred_torch.numpy().flatten()
    y_test_np = y_test.numpy().flatten()

# Cálculo das Métricas
r2_mlp = r2_score(y_test_np, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test_np, y_pred_mlp)
mape_mlp = np.mean(np.abs((y_test_np - y_pred_mlp) / y_test_np)) * 100

print("\n=========================================================")
print("DESEMPENHO DA MLP DE 3 CAMADAS (BENCHMARK INTERNO):")
print("=========================================================")
print(f"R² Score (Acurácia): {r2_mlp:.4f}")
print(f"MAE (Erro Absoluto): {mae_mlp:.4f}")
print(f"MAPE (Erro Percentual): {mape_mlp:.2f}%")
print("=========================================================")

# 6. Gerando Gráfico de Confronto com a MLP
print("\n-> Renderizando o gráfico comparativo com a MLP...")
amostras = 80
plt.figure(figsize=(15, 6))

plt.plot(y_test_np[:amostras], label='Alvo Real (Y_test)', color='#000000', linewidth=2.5)
plt.plot(y_pred_mlp[:amostras], label=f'MLP Tradicional (R²: {r2_mlp:.4f})', color='#9467bd', linewidth=1.8, linestyle='--')

plt.title('Validação Visual: Alvo Real vs. Predição da Rede Neural MLP', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Índice da Amostra de Teste', fontsize=12)
plt.ylabel('Magnitude de Saída (Alvo_Y)', fontsize=12)
plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)

nome_grafico = 'confronto_mlp_vs_real.png'
plt.savefig(nome_grafico, bbox_inches='tight', dpi=300)
plt.close()

print(f"[SUCESSO] Gráfico salvo como: '{nome_grafico}'")
print("Para abrir direto no terminal digite: Start-Process .\\confronto_mlp_vs_real.png")