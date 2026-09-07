import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt

print("=== Inicializando NSGA-II: Otimização Híbrida (Equações + Dados) ===")

# ==============================================================================
# 1. CRIAÇÃO DAS BASES DE DADOS EXPERIMENTAIS (Substituem as fórmulas desconhecidas)
# ==============================================================================

# --- BASE DE DADOS DO OBJETIVO 2 (Ex: Custo/Consumo medido em laboratório) ---
# Matriz de dados coletados para diferentes combinações de x1 (Potência) e x2 (Banda)
pontos_dados_f2 = np.array([
    [0.5, 1.0], [2.0, 1.0], [4.0, 1.0], 
    [0.5, 3.0], [2.0, 3.0], [4.0, 3.0],
    [0.5, 5.0], [2.0, 5.0], [4.0, 5.0]
])
# Respostas reais medidas na bancada para cada ponto acima
valores_dados_f2 = np.array([12.4, 25.1, 48.0, 18.2, 33.7, 58.1, 24.5, 42.9, 72.3])

# Criamos um interpolador 2D: ele lê a base de dados e estima o valor de f2 para qualquer x1 e x2
interpolador_f2 = LinearNDInterpolator(pontos_dados_f2, valores_dados_f2)


# --- BASE DE DADOS DA RESTRIÇÃO 2 (Ex: Limite de Interferência medido por x2) ---
dados_x2_restricao = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
limiares_interferencia = np.array([1.5, 2.8, 4.9, 7.2, 9.8])

# Criamos um interpolador 1D: mapeia o comportamento do limite com base na largura de banda
interpolador_g2 = interp1d(dados_x2_restricao, limiares_interferencia, kind='linear', fill_value="extrapolate")


# ==============================================================================
# 2. CONFIGURAÇÕES DO ALGORITMO GENÉTICO
# ==============================================================================
POP_SIZE = 60
GERACOES = 100
MUT_RATE = 0.15

# Fronteiras físicas das variáveis de decisão [x1, x2, x3]
LB = np.array([0.5, 1.0, 1.0])
UB = np.array([4.0, 5.0, 3.0])

# ==============================================================================
# 3. FUNÇÃO DE AVALIAÇÃO (Casamento entre Equações e Tabelas de Dados)
# ==============================================================================
def avaliar_sistema(x):
    x1, x2, x3 = x[0], x[1], x[2]
    
    # --- OBJETIVO 1 (Baseado em Equação Conhecida) ---
    f1 = (5.0 / (x1 * x2)) + (x3 - 2.2)**2
    
    # --- OBJETIVO 2 (Baseado estritamente na BASE DE DADOS) ---
    # Buscamos a aproximação na tabela de dados + o efeito linear da variável x3
    f2_base_dados = float(interpolador_f2(x1, x2))
    f2 = f2_base_dados + 0.6 * x3
    
    # --- RESTRIÇÃO 1 (Baseada em Equação Conhecida - Limite Térmico) ---
    g1 = 1.2 * x1 + 1.8 * x2 + x3 - 9.5  # Deve ser <= 0 para ser válida
    
    # --- RESTRIÇÃO 2 (Baseada estritamente na BASE DE DADOS - Filtro de Ruído) ---
    limite_da_tabela = float(interpolador_g2(x2))
    g2 = (x1 * x3) - limite_da_tabela  # Deve ser <= 0 para ser válida
    
    # Esquema de Penalização: Se violar as restrições, destrói a nota do indivíduo
    penalidade = 0.0
    if g1 > 0: penalidade += g1 * 500
    if g2 > 0: penalidade += g2 * 500
    
    return [f1 + penalidade, f2 + penalidade]

# ==============================================================================
# 4. FUNÇÕES INTERNAS DO NSGA-II (Dominância e Espalhamento)
# ==============================================================================
def fast_non_dominated_sort(F):
    pop_size = F.shape[0]
    S = [[] for _ in range(pop_size)]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size, dtype=int)
    fronts = [[]]

    for p in range(pop_size):
        for q in range(pop_size):
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1], rank

def crowding_distance(F, front):
    distance = np.zeros(len(front))
    if len(front) <= 2:
        distance[:] = np.inf
        return distance

    for m in range(F.shape[1]):
        obj_values = F[front, m]
        sorted_indices = np.argsort(obj_values)
        distance[sorted_indices[0]] = np.inf
        distance[sorted_indices[-1]] = np.inf
        
        obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
        if obj_range == 0: obj_range = 1e-6
            
        for i in range(1, len(front) - 1):
            distance[sorted_indices[i]] += (obj_values[sorted_indices[i+1]] - obj_values[sorted_indices[i-1]]) / obj_range
    return distance

# ==============================================================================
# 5. LOOP DE EVOLUÇÃO DO ALGORITMO
# ==============================================================================
# População inicial uniforme
X = np.random.uniform(LB, UB, (POP_SIZE, 3))
F = np.array([avaliar_sistema(ind) for ind in X])

for geracao in range(GERACOES):
    X_filhos = []
    while len(X_filhos) < POP_SIZE:
        p1, p2 = X[np.random.choice(POP_SIZE, 2, replace=False)]
        alpha = np.random.rand(3)
        filho = alpha * p1 + (1 - alpha) * p2 # Cruzamento
        
        if np.random.rand() < MUT_RATE:
            filho += np.random.normal(0, 0.1, 3) # Mutação
            
        filho = np.clip(filho, LB, UB)
        X_filhos.append(filho)
        
    X_filhos = np.array(X_filhos)
    F_filhos = np.array([avaliar_sistema(ind) for ind in X_filhos])
    
    X_combinada = np.vstack((X, X_filhos))
    F_combinada = np.vstack((F, F_filhos))
    
    fronts, _ = fast_non_dominated_sort(F_combinada)
    
    novos_indices = []
    for front in fronts:
        if len(novos_indices) + len(front) <= POP_SIZE:
            novos_indices.extend(front)
        else:
            cd = crowding_distance(F_combinada, front)
            sub_indices = np.argsort(cd)[::-1]
            passo = POP_SIZE - len(novos_indices)
            novos_indices.extend([front[idx] for idx in sub_indices[:passo]])
            break
            
    X = X_combinada[novos_indices]
    F = F_combinada[novos_indices]

# Isolar a melhor casca de soluções válidas (Fronteira de Pareto)
fronts, _ = fast_non_dominated_sort(F)
fronteira_final = F[fronts[0]]

# ==============================================================================
# 6. GERAÇÃO E SALVAMENTO DO GRÁFICO RESIDUAL
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.scatter(fronteira_final[:, 0], fronteira_final[:, 1], color='#10b981', edgecolors='k', s=50, label='Soluções Ótimas (Pareto)')
plt.title('Fronteira de Otimização: Equações vs. Bases de Dados', fontsize=11, fontweight='bold')
plt.xlabel('f1: Métrica via Equação (Mínimo)')
plt.ylabel('f2: Métrica via Base de Dados (Mínimo)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('pareto_hibrido_dados.png', dpi=300, bbox_inches='tight')
plt.close()
print("[CONCLUÍDO] Gráfico gerado com sucesso: 'pareto_hibrido_dados.png'")