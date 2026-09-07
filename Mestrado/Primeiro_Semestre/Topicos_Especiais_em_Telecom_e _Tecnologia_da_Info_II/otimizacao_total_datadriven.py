import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("=== CONSTRUINDO UM SISTEMA 100% DATA-DRIVEN (UFES 2026) ===")

# ==============================================================================
# 1. GERANDO AS BASES DE DADOS DE BANCADA (SIMULAÇÃO DE COLETA DO LABORATÓRIO)
# ==============================================================================
# Amostras coletadas para x1 (Potência) e x2 (Banda). Mantendo x3 fixo em 2.2 para os dados
X_bancada = np.array([
    [0.5, 1.0], [2.0, 1.0], [4.0, 1.0], 
    [0.5, 3.0], [2.0, 3.0], [4.0, 3.0],
    [0.5, 5.0], [2.0, 5.0], [4.0, 5.0]
], dtype=np.float32)

# Dados coletados em campo para o Objetivo 1 (Erro de sinal medido)
y_dados_f1 = np.array([[10.05], [2.52], [1.28], [3.35], [0.85], [0.45], [2.02], [0.52], [0.28]], dtype=np.float32)

# Dados coletados em campo para o Objetivo 2 (Custo medido)
y_dados_f2 = np.array([[12.4], [25.1], [48.0], [18.2], [33.7], [58.1], [24.5], [42.9], [72.3]], dtype=np.float32)

# ==============================================================================
# 2. OBJETIVO 1: GERANDO A EQUAÇÃO VIA REGRESSÃO POLINOMIAL (DATA-DRIVEN)
# ==============================================================================
# Criamos características polinomiais de grau 2 (x1^2, x2^2, x1*x2, etc.)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_bancada)

modelo_regressao = LinearRegression()
modelo_regressao.fit(X_poly, y_dados_f1)

coef = modelo_regressao.coef_[0]
c_inter = modelo_regressao.intercept_[0]

print("\n--> EQUAÇÃO DATA-DRIVEN GERADA PARA O OBJETIVO 1:")
print(f"f1(x1, x2) = {c_inter:.3f} + ({coef[0]:.3f})*x1 + ({coef[1]:.3f})*x2 + ({coef[2]:.3f})*x1^2 + ({coef[3]:.3f})*x1*x2 + ({coef[4]:.3f})*x2^2")

# ==============================================================================
# 3. OBJETIVO 2: TREINANDO A REDE NEURAL MLP (DATA-DRIVEN)
# ==============================================================================
X_tensor = torch.tensor(X_bancada)
y_tensor_f2 = torch.tensor(y_dados_f2)

class RedeSubstituta(nn.Module):
    def __init__(self):
        super(RedeSubstituta, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

modelo_mlp = RedeSubstituta()
criterion = nn.MSELoss()
optimizer = optim.Adam(modelo_mlp.parameters(), lr=0.01)

# Treino rápido da rede caixa-preta
modelo_mlp.train()
for epoch in range(1200):
    optimizer.zero_grad()
    loss = criterion(modelo_mlp(X_tensor), y_tensor_f2)
    loss.backward()
    optimizer.step()
modelo_mlp.eval()
print(f"   [OK] Rede Neural treinada para o Objetivo 2. Erro Final: {loss.item():.6f}")

# ==============================================================================
# 4. AVALIAÇÃO DA POPULAÇÃO USANDO APENAS OS MODELOS APRENDIDOS
# ==============================================================================
# Limites operacionais das variáveis
LB = np.array([0.5, 1.0])
UB = np.array([4.0, 5.0])
POP_SIZE = 40
GERACOES = 60
MUT_RATE = 0.1

def avaliar_total_datadriven(x):
    x1, x2 = x[0], x[1]
    
    # f1 calculado pela EQUAÇÃO POLINOMIAL GERADA PELOS DADOS
    X_f = poly.transform(np.array([[x1, x2]]))
    f1 = float(modelo_regressao.predict(X_f)[0][0])
    
    # f2 calculado pela PREDICAO DA REDE NEURAL
    entrada_torch = torch.tensor([[x1, x2]], dtype=torch.float32)
    with torch.no_grad():
        f2 = float(modelo_mlp(entrada_torch).item())
        
    # Restrição de segurança simples (Exemplo: Potência + Banda acumulada não pode estourar 7.5)
    g1 = x1 + x2 - 7.5
    penalidade = 0.0
    if g1 > 0: penalidade += g1 * 500
        
    return f1, f2, penalidade

# ==============================================================================
# 5. ALGORITMO GENÉTICO MONOOBJETIVO PONDERADO
# ==============================================================================
def otimizar_pesos(alpha):
    X = np.random.uniform(LB, UB, (POP_SIZE, 2))
    for gen in range(GERACOES):
        fitness = []
        for ind in X:
            f1, f2, pen = avaliar_total_datadriven(ind)
            fitness.append(alpha * f1 + (1 - alpha) * f2 + pen)
            
        X = X[np.argsort(fitness)]
        X_filhos = []
        while len(X_filhos) < POP_SIZE // 2:
            idx_p1, idx_p2 = np.random.choice(POP_SIZE // 2, 2, replace=False)
            w = np.random.rand(2)
            filho = np.clip(w * X[idx_p1] + (1 - w) * X[idx_p2], LB, UB)
            if np.random.rand() < MUT_RATE:
                filho = np.clip(filho + np.random.normal(0, 0.05, 2), LB, UB)
            X_filhos.append(filho)
        X[POP_SIZE // 2:] = np.array(X_filhos)
        
    f1_otimo, f2_otimo, _ = avaliar_total_datadriven(X[0])
    return f1_otimo, f2_otimo

# Varredura dos coeficientes de Alpha
valores_alpha = np.linspace(0.001, 0.999, 50)
fronteira_f1 = []
fronteira_f2 = []

for alpha in valores_alpha:
    f1, f2 = otimizar_pesos(alpha)
    fronteira_f1.append(f1)
    fronteira_f2.append(f2)

# Ordenação dos pontos para evitar linhas cruzadas no plot
indices_ordenados = np.argsort(fronteira_f1)
fronteira_f1 = np.array(fronteira_f1)[indices_ordenados]
fronteira_f2 = np.array(fronteira_f2)[indices_ordenados]

# ==============================================================================
# 6. PLOT DA FRONTEIRA TOTALMENTE DATA-DRIVEN
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.plot(fronteira_f1, fronteira_f2, color='#db2777', linestyle='-', alpha=0.4)
plt.scatter(fronteira_f1, fronteira_f2, color='#be185d', edgecolors='k', s=50, label='Pareto: 100% Data-Driven')
plt.title('Fronteira de Pareto: Equação Polinomial vs. Rede Neural', fontsize=11, fontweight='bold')
plt.xlabel('f1: Erro Estimado pela Equação de Regressão (Mínimo)')
plt.ylabel('f2: Custo Estimado pela Rede Neural (Mínimo)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('pareto_total_datadriven.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SUCESSO] Otimização concluída! Gráfico salvo como 'pareto_total_datadriven.png'")