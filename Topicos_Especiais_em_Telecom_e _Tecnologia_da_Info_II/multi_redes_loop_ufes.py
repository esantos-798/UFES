import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print("=== SISTEMA MULTI-REDES COM LOOP DE OTIMIZADORES (UFES 2026) ===")

# ==============================================================================
# 1. BASES DE DADOS DE BANCADA (AMOSTRAGENS DO LABORATÓRIO)
# ==============================================================================
# Entradas comuns para ambas as redes: [x1 (Potência), x2 (Banda)]
X_bancada = np.array([
    [0.5, 1.0], [2.0, 1.0], [4.0, 1.0], 
    [0.5, 3.0], [2.0, 3.0], [4.0, 3.0],
    [0.5, 5.0], [2.0, 5.0], [4.0, 5.0]
], dtype=np.float32)

# Alvos coletados para o Objetivo 1 (ex: Perda de Pacotes)
y_dados_f1 = np.array([[10.05], [2.52], [1.28], [3.35], [0.85], [0.45], [2.02], [0.52], [0.28]], dtype=np.float32)

# Alvos coletados para o Objetivo 2 (ex: Custo de Infraestrutura)
y_dados_f2 = np.array([[12.4], [25.1], [48.0], [18.2], [33.7], [58.1], [24.5], [42.9], [72.3]], dtype=np.float32)

X_tensor = torch.tensor(X_bancada)
y_tensor_f1 = torch.tensor(y_dados_f1)
y_tensor_f2 = torch.tensor(y_dados_f2)

# ==============================================================================
# 2. ARQUITETURA PADRÃO DA REDE NEURAL (MLP REGRESSORA)
# ==============================================================================
class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 3. LOOP DE SELEÇÃO DE OTIMIZADORES PARA AS DUAS REDES
# ==============================================================================
otimizadores_para_testar = ['Adam', 'SGD', 'RMSprop']

def treinar_e_selecionar_melhor_rede(X_in, y_target, nome_objetivo):
    melhor_loss = float('inf')
    melhor_modelo = None
    melhor_otimizador_nome = ""
    
    print(f"\n-> Iniciando loop de otimizadores para a rede do {nome_objetivo}:")
    
    for opt_nome in otimizadores_para_testar:
        modelo = MLPRegressor()
        criterion = nn.MSELoss()
        
        # Seleção dinâmica do otimizador via loop
        if opt_nome == 'Adam':
            optimizer = optim.Adam(modelo.parameters(), lr=0.01)
        elif opt_nome == 'SGD':
            optimizer = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9)
        elif opt_nome == 'RMSprop':
            optimizer = optim.RMSprop(modelo.parameters(), lr=0.005)
            
        # Treinamento
        modelo.train()
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = criterion(modelo(X_in), y_target)
            loss.backward()
            optimizer.step()
            
        loss_final = loss.item()
        print(f"   Otimizador: {opt_nome} | Erro Final (MSE): {loss_final:.6f}")
        
        if loss_final < melhor_loss:
            melhor_loss = loss_final
            melhor_modelo = modelo
            melhor_otimizador_nome = opt_nome
            
    print(f"   >> Campeão para {nome_objetivo}: {melhor_otimizador_nome} (MSE: {melhor_loss:.6f})")
    melhor_modelo.eval()
    return melhor_modelo

# Treinando e salvando as melhores instâncias de redes
modelo_f1_campeao = treinar_e_selecionar_melhor_rede(X_tensor, y_tensor_f1, "Objetivo 1")
modelo_f2_campeao = treinar_e_selecionar_melhor_rede(X_tensor, y_tensor_f2, "Objetivo 2")

# ==============================================================================
# 4. AVALIAÇÃO DA POPULAÇÃO BASEADA NAS DUAS REDES NEURAIS
# ==============================================================================
LB = np.array([0.5, 1.0])
UB = np.array([4.0, 5.0])
POP_SIZE = 45
GERACOES = 70
MUT_RATE = 0.12

def avaliar_duas_redes(x):
    x1, x2 = x[0], x[1]
    entrada_torch = torch.tensor([[x1, x2]], dtype=torch.float32)
    
    with torch.no_grad():
        # f1 estimado pela primeira Rede Neural campeã
        f1 = float(modelo_f1_campeao(entrada_torch).item())
        # f2 estimado pela segunda Rede Neural campeã
        f2 = float(modelo_mlp_campeao_f2 := modelo_f2_campeao(entrada_torch).item())
        
    # Restrição física do acoplamento
    g1 = x1 + x2 - 7.5
    penalidade = 0.0
    if g1 > 0: penalidade += g1 * 600
        
    return f1, f2, penalidade

# ==============================================================================
# 5. ALGORITMO GENÉTICO MONOOBJETIVO PONDERADO
# ==============================================================================
def otimizar_com_multi_redes(alpha):
    X = np.random.uniform(LB, UB, (POP_SIZE, 2))
    for gen in range(GERACOES):
        fitness = []
        for ind in X:
            f1, f2, pen = avaliar_duas_redes(ind)
            fitness.append(alpha * f1 + (1 - alpha) * f2 + pen)
            
        X_filhos = []
        vagas_restantes = POP_SIZE - (POP_SIZE // 2) # Vai dar 23 em vez de 22
        while len(X_filhos) < vagas_restantes:
            idx_p1, idx_p2 = np.random.choice(POP_SIZE // 2, 2, replace=False)
            w = np.random.rand(2)
            filho = np.clip(w * X[idx_p1] + (1 - w) * X[idx_p2], LB, UB)
            if np.random.rand() < MUT_RATE:
                filho = np.clip(filho + np.random.normal(0, 0.05, 2), LB, UB)
            X_filhos.append(filho)
        X[POP_SIZE // 2:] = np.array(X_filhos)
        
    f1_otimo, f2_otimo, _ = avaliar_duas_redes(X[0])
    return f1_otimo, f2_otimo

# Varredura fina de pesos Alpha para desenhar a curva
valores_alpha = np.linspace(0.001, 0.999, 55)
fronteira_f1 = []
fronteira_f2 = []

print("\n-> Varrendo a Fronteira de Pareto com as duas Redes Neurais integradas...")
for alpha in valores_alpha:
    f1, f2 = otimizar_com_multi_redes(alpha)
    fronteira_f1.append(f1)
    fronteira_f2.append(f2)

# Ordenação para polimento gráfico
indices_ordenados = np.argsort(fronteira_f1)
fronteira_f1 = np.array(fronteira_f1)[indices_ordenados]
fronteira_f2 = np.array(fronteira_f2)[indices_ordenados]

# ==============================================================================
# 6. PLOT DA FRONTEIRA INTEIRAMENTE BASEADA EM I.A.
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.plot(fronteira_f1, fronteira_f2, color='#10b981', linestyle='-', alpha=0.4)
plt.scatter(fronteira_f1, fronteira_f2, color='#059669', edgecolors='k', s=50, label='Pareto: MLP(F1) vs. MLP(F2)')
plt.title('Fronteira de Pareto Otimizada por Duas Redes Neurais (PyTorch)', fontsize=11, fontweight='bold')
plt.xlabel('f1: Erro Estimado pela MLP 1 (Mínimo)')
plt.ylabel('f2: Custo Estimado pela MLP 2 (Mínimo)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('pareto_multi_redes.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n[SUCESSO] Processo finalizado! Gráfico salvo como 'pareto_multi_redes.png'")