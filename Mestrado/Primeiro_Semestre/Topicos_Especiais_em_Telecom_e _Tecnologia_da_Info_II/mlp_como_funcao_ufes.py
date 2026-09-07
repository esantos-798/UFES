import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

print("=== TREINANDO A REDE NEURAL COMO UMA FUNÇÃO OBJETIVO ===")

# ==============================================================================
# 1. BASE DE DADOS DO OBJETIVO 2 (CUSTOS MEDIDOS)
# ==============================================================================
# Entradas: [x1 (Potência), x2 (Banda)]
X_dados = np.array([
    [0.5, 1.0], [2.0, 1.0], [4.0, 1.0], 
    [0.5, 3.0], [2.0, 3.0], [4.0, 3.0],
    [0.5, 5.0], [2.0, 5.0], [4.0, 5.0]
], dtype=np.float32)

# Saídas: Custo medido em bancada
y_dados = np.array([[12.4], [25.1], [48.0], [18.2], [33.7], [58.1], [24.5], [42.9], [72.3]], dtype=np.float32)

# Conversão para tensores do PyTorch
X_train = torch.tensor(X_dados)
y_train = torch.tensor(y_dados)

# ==============================================================================
# 2. ARQUITETURA DA REDE NEURAL (APROXIMADOR DE FUNÇÃO)
# ==============================================================================
class RedeFuncao(nn.Module):
    def __init__(self):
        super(RedeFuncao, self).__init__()
        self.camadas = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Saída contínua (Regressão do Custo)
        )
    def forward(self, x):
        return self.camadas(x)

# Treinamento rápido da Rede Caixa-Preta
modelo_f2 = RedeFuncao()
criterion = nn.MSELoss()
optimizer = optim.Adam(modelo_f2.parameters(), lr=0.01)

print("-> Treinando a Rede Neural para mapear a base de dados...")
modelo_f2.train()
for epoch in range(1500):
    optimizer.zero_grad()
    outputs = modelo_f2(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

print(f"   [OK] Treinamento concluído. Erro Final (MSE): {loss.item():.6f}")
modelo_f2.eval()

# ==============================================================================
# 3. BASE DE DADOS CONHECIDA DA RESTRIÇÃO 2 (INTERPOLADOR 1D)
# ==============================================================================
dados_x2_restricao = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
limiares_interferencia = np.array([1.5, 2.8, 4.9, 7.2, 9.8])
interpolador_g2 = interp1d(dados_x2_restricao, limiares_interferencia, kind='linear', fill_value="extrapolate")

# Configurações do Algoritmo Genético
POP_SIZE = 40
GERACOES = 60
MUT_RATE = 0.1
LB = np.array([0.5, 1.0, 1.0])
UB = np.array([4.0, 5.0, 3.0])

# ==============================================================================
# 4. AVALIAÇÃO DO SISTEMA USANDO A REDE NEURAL NO LOOP
# ==============================================================================
def avaliar_com_rede(x):
    x1, x2, x3 = x[0], x[1], x[2]
    
    # f1: Calculado via Equação Analítica
    f1 = (5.0 / (x1 * x2)) + (x3 - 2.2)**2
    
    # f2: CALCULADO PELA REDE NEURAL TREINADA!
    entrada_rede = torch.tensor([[x1, x2]], dtype=torch.float32)
    with torch.no_grad():
        f2_previsto_rede = float(modelo_f2(entrada_rede).item())
    
    f2 = f2_previsto_rede + 0.6 * x3
    
    # Restrições (g1: Equação, g2: Dados)
    g1 = 1.2 * x1 + 1.8 * x2 + x3 - 9.5
    g2 = (x1 * x3) - float(interpolador_g2(x2))
    
    penalidade = 0.0
    if g1 > 0: penalidade += g1 * 1000
    if g2 > 0: penalidade += g2 * 1000
        
    return f1, f2, penalidade

# ==============================================================================
# 5. ALGORITMO GENÉTICO MONOOBJETIVO PONDERADO
# ==============================================================================
def otimizar_com_rede(alpha):
    X = np.random.uniform(LB, UB, (POP_SIZE, 3))
    for gen in range(GERACOES):
        fitness = []
        for ind in X:
            f1, f2, pen = avaliar_com_rede(ind)
            fitness.append(alpha * f1 + (1 - alpha) * f2 + pen)
        
        X = X[np.argsort(fitness)]
        X_filhos = []
        while len(X_filhos) < POP_SIZE // 2:
            idx_p1, idx_p2 = np.random.choice(POP_SIZE // 2, 2, replace=False)
            w = np.random.rand(3)
            filho = np.clip(w * X[idx_p1] + (1 - w) * X[idx_p2], LB, UB)
            if np.random.rand() < MUT_RATE:
                filho = np.clip(filho + np.random.normal(0, 0.05, 3), LB, UB)
            X_filhos.append(filho)
        X[POP_SIZE // 2:] = np.array(X_filhos)
        
    f1_otimo, f2_otimo, _ = avaliar_com_rede(X[0])
    return f1_otimo, f2_otimo

# Varredura de pesos para construir a Fronteira de Pareto
valores_alpha = np.linspace(0.001, 0.999, 50)
fronteira_f1 = []
fronteira_f2 = []

print("-> Executando o Algoritmo Genético usando a Rede Neural como avaliadora...")
for alpha in valores_alpha:
    f1, f2 = otimizar_com_rede(alpha)
    fronteira_f1.append(f1)
    fronteira_f2.append(f2)

# ==============================================================================
# 6. PLOTAGEM DO RESULTADO FINAL
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.plot(fronteira_f1, fronteira_f2, color='#f59e0b', linestyle='-', alpha=0.3)
plt.scatter(fronteira_f1, fronteira_f2, color='#d97706', edgecolors='k', s=50, label='Pareto (F2 via Rede Neural)')
plt.title('Fronteira de Pareto: Objetivo Substituído por Rede Neural (2026)', fontsize=11, fontweight='bold')
plt.xlabel('f1: Métrica via Equação (Mínimo)')
plt.ylabel('f2: Métrica via Predição da MLP (Mínimo)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('pareto_via_rede_neural.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SUCESSO] Gráfico atualizado salvo como 'pareto_via_rede_neural.png'")