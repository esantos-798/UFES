import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. BENCHMARK MULTI-OBJETIVO (ZDT1) E AUXILIARES DE PARETO
# =============================================================================
def zdt1(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2])

def domina(a_costs, b_costs):
    return np.all(a_costs <= b_costs) and np.any(a_costs < b_costs)

def calcular_crowding_distance(archive_costs):
    n = len(archive_costs)
    if n == 0: return np.array([])
    if n <= 2: return np.inf * np.ones(n)
    
    num_objs = archive_costs.shape[1]
    distances = np.zeros(n)
    
    for m in range(num_objs):
        idx = np.argsort(archive_costs[:, m])
        distances[idx[0]] = np.inf
        distances[idx[-1]] = np.inf
        
        obj_min = archive_costs[idx[0], m]
        obj_max = archive_costs[idx[-1], m]
        norm = obj_max - obj_min if obj_max != obj_min else 1.0
        
        for i in range(1, n - 1):
            distances[idx[i]] += (archive_costs[idx[i+1], m] - archive_costs[idx[i-1], m]) / norm
            
    return distances

def selecionar_lider_roleta(archive_pos, archive_costs):
    # Calcula as distâncias. Valores infinitos (fronteiras) ganham peso alto estável
    dist = calcular_crowding_distance(archive_costs)
    dist[np.isinf(dist)] = np.max(dist[~np.isinf(dist)]) * 2 if np.any(~np.isinf(dist)) else 1.0
    
    # Roleta baseada na distância de aglomeração (privilegia os mais isolados)
    probs = dist / np.sum(dist)
    idx_escolhido = np.random.choice(len(archive_pos), p=probs)
    return archive_pos[idx_escolhido]

# =============================================================================
# 2. ALGORITMO MOHAF PRINCIPAL
# =============================================================================
def run_mohaf(max_archive=50):
    D = 30
    num_pop = 30
    max_iter = 100
    bounds = [0.0, 1.0] # Domínio padrão do ZDT1
    
    # Inicialização da População
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    X_costs = np.array([zdt1(ind) for ind in X])
    
    # Inicialização do Arquivo Externo (Armazena as Soluções Não-Dominadas)
    archive_pos = []
    archive_costs = []
    
    # Loop de Otimização Multi-Objetivo
    for t in range(max_iter):
        
        # --- ATUALIZAÇÃO DO ARQUIVO EXTERNO (PARETO) ---
        for i in range(num_pop):
            sol_dominada = False
            eliminar_do_archive = []
            
            for j in range(len(archive_pos)):
                if domina(archive_costs[j], X_costs[i]):
                    sol_dominada = True
                    break
                elif domina(X_costs[i], archive_costs[j]):
                    eliminar_do_archive.append(j)
            
            if not sol_dominada:
                # Remove quem a nova solução dominou de trás para frente
                for idx in sorted(eliminar_do_archive, reverse=True):
                    archive_pos.pop(idx)
                    archive_costs.pop(idx)
                archive_pos.append(np.copy(X[i]))
                archive_costs.append(np.copy(X_costs[i]))
        
        # Poda do Arquivo se estourar o limite (Mantém os de maior Crowding Distance)
        if len(archive_pos) > max_archive:
            dist = calcular_crowding_distance(np.array(archive_costs))
            idx_remover = np.argsort(dist)[:(len(archive_pos) - max_archive)]
            for idx in sorted(idx_remover, reverse=True):
                archive_pos.pop(idx)
                archive_costs.pop(idx)
                
        # --- MOVIMENTAÇÃO DO FLUIDO (MOTOR HAF ULTRA CORRIGIDO) ---
        a = 2.0 * np.log(1.0 + (max_iter - t) / max_iter) / np.log(2.0)
        
        for i in range(num_pop):
            # Seleciona probabilisticamente o ralo atrator de dentro do arquivo de Pareto
            best_pos = selecionar_lider_roleta(archive_pos, np.array(archive_costs))
            
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2 * np.cos(np.pi * (t / max_iter))
            X_vizinho = X[np.random.randint(0, num_pop)]
            
            # Operador Geométrico de Escoamento
            if np.abs(A[0]) > 1.0:
                X[i] = X_vizinho - A * np.abs(C * X_vizinho - X[i])
            else:
                X[i] = best_pos - A * np.abs(C * best_pos - X[i])
                
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            X_costs[i] = zdt1(X[i])
            
    return np.array(archive_costs)

# =============================================================================
# 3. EXECUÇÃO E VISUALIZAÇÃO DA FRONTEIRA DE PARETO
# =============================================================================
print("Executando o MOHAF v1.0 no benchmark ZDT1...")
fronteira_obtida = run_mohaf(max_archive=60)

# Curva teórica perfeita do ZDT1 para comparação visual
f1_teorico = np.linspace(0, 1, 200)
f2_teorico = 1.0 - np.sqrt(f1_teorico)

plt.figure(figsize=(8, 6))
plt.plot(f1_teorico, f2_teorico, color='red', linestyle='--', label='Fronteira Teórica de Pareto')
plt.scatter(fronteira_obtida[:, 0], fronteira_obtida[:, 1], color='blue', edgecolors='k', s=45, label='MOHAF (Arquivo Externo)')
plt.title('Fronteira de Pareto Obtida pelo MOHAF (ZDT1)', fontsize=12, fontweight='bold')
plt.xlabel('Objetivo 1 (f1)')
plt.ylabel('Objetivo 2 (f2)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()