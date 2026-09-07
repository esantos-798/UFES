import numpy as np
import matplotlib.pyplot as plt

# Bibliotecas de estado da arte para os competidores
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

# =============================================================================
# 1. SEU MOTOR: MOHAF v1.0 (ZDT1)
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
    if n <= 2: return np.inf * np.ones(n)
    num_objs = archive_costs.shape[1]
    distances = np.zeros(n)
    for m in range(num_objs):
        idx = np.argsort(archive_costs[:, m])
        distances[idx[0]] = np.inf
        distances[idx[-1]] = np.inf
        obj_min, obj_max = archive_costs[idx[0], m], archive_costs[idx[-1], m]
        norm = obj_max - obj_min if obj_max != obj_min else 1.0
        for i in range(1, n - 1):
            distances[idx[i]] += (archive_costs[idx[i+1], m] - archive_costs[idx[i-1], m]) / norm
    return distances

def selecionar_lider_roleta(archive_pos, archive_costs):
    dist = calcular_crowding_distance(archive_costs)
    dist[np.isinf(dist)] = np.max(dist[~np.isinf(dist)]) * 2 if np.any(~np.isinf(dist)) else 1.0
    probs = dist / np.sum(dist)
    return archive_pos[np.random.choice(len(archive_pos), p=probs)]

def run_mohaf(max_archive=60):
    D, num_pop, max_iter = 30, 30, 100
    bounds = [0.0, 1.0]
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    X_costs = np.array([zdt1(ind) for ind in X])
    archive_pos, archive_costs = [], []
    
    for t in range(max_iter):
        for i in range(num_pop):
            sol_dominada = False
            eliminar_do_archive = []
            for j in range(len(archive_pos)):
                if domina(archive_costs[j], X_costs[i]):
                    sol_dominada = True; break
                elif domina(X_costs[i], archive_costs[j]):
                    eliminar_do_archive.append(j)
            if not sol_dominada:
                for idx in sorted(eliminar_do_archive, reverse=True):
                    archive_pos.pop(idx); archive_costs.pop(idx)
                archive_pos.append(np.copy(X[i])); archive_costs.append(np.copy(X_costs[i]))
        if len(archive_pos) > max_archive:
            dist = calcular_crowding_distance(np.array(archive_costs))
            idx_remover = np.argsort(dist)[:(len(archive_pos) - max_archive)]
            for idx in sorted(idx_remover, reverse=True):
                archive_pos.pop(idx); archive_costs.pop(idx)
        
        a = 2.0 * np.log(1.0 + (max_iter - t) / max_iter) / np.log(2.0)
        for i in range(num_pop):
            best_pos = selecionar_lider_roleta(archive_pos, np.array(archive_costs))
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A, C = 2.0 * a * r1 - a, 2.0 * r2 * np.cos(np.pi * (t / max_iter))
            X_vizinho = X[np.random.randint(0, num_pop)]
            X[i] = X_vizinho - A * np.abs(C * X_vizinho - X[i]) if np.abs(A[0]) > 1.0 else best_pos - A * np.abs(C * best_pos - X[i])
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            X_costs[i] = zdt1(X[i])
            
    return np.array(archive_costs)

# =============================================================================
# 2. EXECUÇÃO COMPARATIVA (MOHAF vs NSGA-III vs MOEA/D)
# =============================================================================
print("1/3 - Executando o seu MOHAF...")
mohaf_res = run_mohaf(max_archive=50)

# Configuração do problema ZDT1 padrão na pymoo (D=30, 2 objetivos)
problem = get_problem("zdt1")

# Configurando direções de referência para o NSGA-III e MOEA/D (população próxima de 30)
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=30)

print("2/3 - Executando o NSGA-III de referência...")
algo_nsga3 = NSGA3(ref_dirs=ref_dirs)
res_nsga3 = minimize(problem, algo_nsga3, ('n_gen', 100), seed=1, verbose=False)

print("3/3 - Executando o MOEA/D de referência...")
algo_moead = MOEAD(ref_dirs=ref_dirs, n_neighbors=15)
res_moead = minimize(problem, algo_moead, ('n_gen', 100), seed=1, verbose=False)

# =============================================================================
# 3. VISUALIZAÇÃO DO EMBATE CIENTÍFICO
# =============================================================================
f1_teorico = np.linspace(0, 1, 200)
f2_teorico = 1.0 - np.sqrt(f1_teorico)

plt.figure(figsize=(9, 7))
plt.plot(f1_teorico, f2_teorico, color='black', linestyle='--', alpha=0.7, label='Fronteira Teórica')

# Plot das 3 meta-heurísticas
plt.scatter(mohaf_res[:, 0], mohaf_res[:, 1], color='blue', edgecolors='k', s=50, label='MOHAF (Seu Algoritmo)', zorder=3)
plt.scatter(res_nsga3.F[:, 0], res_nsga3.F[:, 1], color='orange', marker='s', edgecolors='k', s=40, label='NSGA-III', alpha=0.8)
plt.scatter(res_moead.F[:, 0], res_moead.F[:, 1], color='green', marker='^', edgecolors='k', s=40, label='MOEA/D', alpha=0.8)

plt.title('Batalha de Meta-heurísticas: MOHAF vs Estado da Arte (ZDT1)', fontsize=12, fontweight='bold')
plt.xlabel('Objetivo 1 (f1)')
plt.ylabel('Objetivo 2 (f2)')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.show()