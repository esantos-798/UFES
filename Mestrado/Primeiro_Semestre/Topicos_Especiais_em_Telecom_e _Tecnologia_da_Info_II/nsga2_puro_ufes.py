import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt

print("=== Inicializando NSGA-II Construído do Zero (Sem Pymoo) ===")

# ==============================================================================
# 1. LOOKUP TABLES (DADOS EXPERIMENTAIS DO LABORATÓRIO)
# ==============================================================================
# Dados para o Objetivo 2: Custo baseado em x1 (potência) e x2 (banda)
pontos_x1_x2 = np.array([[0.5, 1.0], [2.0, 1.0], [4.0, 1.0], 
                         [0.5, 5.0], [2.0, 5.0], [4.0, 5.0]])
valores_custo = np.array([10.5, 22.1, 45.0, 15.3, 30.2, 60.8])
lookup_f2_dados = LinearNDInterpolator(pontos_x1_x2, valores_custo)

# Dados para a Restrição 2: Limiar de interferência baseado em x2 (banda)
dados_banda_x2 = np.array([1.0, 2.5, 4.0, 5.0])
dados_interferencia = np.array([1.2, 3.5, 6.8, 9.5])
lookup_g2_dados = interp1d(dados_banda_x2, dados_interferencia, kind='linear', fill_value="extrapolate")

# Parâmetros do Algoritmo
POP_SIZE = 60
GERACOES = 80
MUT_RATE = 0.15

# Limites das 3 Variáveis de Decisão [x1, x2, x3]
LB = np.array([0.5, 1.0, 1.0])
UB = np.array([4.0, 5.0, 3.0])

# ==============================================================================
# 2. FUNÇÃO DE AVALIAÇÃO (MISTURA DE EQUAÇÕES E DADOS)
# ==============================================================================
def avaliar_individuo(x):
    x1, x2, x3 = x[0], x[1], x[2]
    
    # Objetivo 1 (Equação) e Objetivo 2 (Dados)
    f1 = (4.0 / (x1 * x2)) + (x3 - 2.5)**2
    f2 = float(lookup_f2_dados(x1, x2)) + 0.5 * x3
    
    # Restrição 1 (Equação) e Restrição 2 (Dados)
    g1 = 1.5 * x1 + 2.0 * x2 + x3 - 10.0
    g2 = (x1 * x3) - float(lookup_g2_dados(x2))
    
    # Penalização por Restrição (Se quebrar a restrição, piora artificialmente o objetivo)
    penalidade = 0.0
    if g1 > 0: penalidade += g1 * 100
    if g2 > 0: penalidade += g2 * 100
    
    return [f1 + penalidade, f2 + penalidade]

# ==============================================================================
# 3. OPERADORES DO ALGORITMO GENÉTICO MULTIOBJETIVO
# ==============================================================================
def fast_non_dominated_sort(F):
    """Agrupa os indivíduos por Fronteiras de Pareto (Ranks)"""
    pop_size = F.shape[0]
    S = [[] for _ in range(pop_size)]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size, dtype=int)
    fronts = [[]]

    for p in range(pop_size):
        for q in range(pop_size):
            # Condição de dominância de Pareto (Minimização)
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
    """Calcula o espalhamento para evitar agrupamento excessivo"""
    distance = np.zeros(len(front))
    if len(front) <= 2:
        distance[:] = np.inf
        return distance

    for m in range(F.shape[1]): # Para cada objetivo
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
# 4. LOOP PRINCIPAL DA EVOLUÇÃO
# ==============================================================================
# Gerar População Inicial Aleatória respeitando as fronteiras
X = np.random.uniform(LB, UB, (POP_SIZE, 3))
F = np.array([avaliar_individuo(ind) for ind in X])

for geracao in range(GERACOES):
    # Gerar descendentes (Cruzamento e Mutação)
    X_filhos = []
    while len(X_filhos) < POP_SIZE:
        # Seleção por Torneio Simples
        p1, p2 = X[np.random.choice(POP_SIZE, 2, replace=False)]
        # Cruzamento Aritmético (Blended Crossover)
        alpha = np.random.rand(3)
        filho = alpha * p1 + (1 - alpha) * p2
        # Mutação Gaussiana
        if np.random.rand() < MUT_RATE:
            filho += np.random.normal(0, 0.1, 3)
        # Clipar dentro dos limites de fronteira das variáveis
        filho = np.clip(filho, LB, UB)
        X_filhos.append(filho)
        
    X_filhos = np.array(X_filhos)
    F_filhos = np.array([avaliar_individuo(ind) for ind in X_filhos])
    
    # Combinar Pais e Filhos (Tamanho 2 * POP_SIZE)
    X_combinada = np.vstack((X, X_filhos))
    F_combinada = np.vstack((F, F_filhos))
    
    # Classificar a população combinada
    fronts, ranks = fast_non_dominated_sort(F_combinada)
    
    # Selecionar os melhores para a próxima geração
    novos_indices = []
    for front in fronts:
        if len(novos_indices) + len(front) <= POP_SIZE:
            novos_indices.extend(front)
        else:
            # Se a fronteira quebra o limite, trunca por Crowding Distance
            cd = crowding_distance(F_combinada, front)
            sub_indices = np.argsort(cd)[::-1] # Maiores distâncias primeiro
            passo = POP_SIZE - len(novos_indices)
            novos_indices.extend([front[idx] for idx in sub_indices[:passo]])
            break
            
    X = X_combinada[novos_indices]
    F = F_combinada[novos_indices]

# Filtrar apenas a primeira Fronteira de Pareto real para o gráfico
fronts, _ = fast_non_dominated_sort(F)
fronteira_final_F = F[fronts[0]]

# ==============================================================================
# 5. GRAFICO DA FRONTEIRA DE PARETO PURO
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.scatter(fronteira_final_F[:, 0], fronteira_final_F[:, 1], color='#e11d48', edgecolors='k', s=45, label='Fronteira Ótima (Pareto)')
plt.title('NSGA-II Construído do Zero (UFES 2026)', fontsize=11, fontweight='bold')
plt.xlabel('f1: Erro de Sinal (Equação - Mínimo)')
plt.ylabel('f2: Custo Operacional (Dados - Mínimo)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('pareto_puro_sem_pymoo.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SUCESSO] Otimização concluída e gráfico salvo como 'pareto_puro_sem_pymoo.png'")