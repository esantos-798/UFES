import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

# =============================================================================
# OPERADORES E MOTOR MOHAF (MANTENDO A LÓGICA DO SEU ARQUIVO)
# =============================================================================
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
    if np.all(np.isinf(dist)):
        probs = np.ones(len(archive_pos)) / len(archive_pos)
    else:
        valores_reais = dist[~np.isinf(dist)]
        max_real = np.max(valores_reais) if len(valores_reais) > 0 else 1.0
        dist[np.isinf(dist)] = max_real * 2.0
        dist = np.maximum(dist, 1e-6)
        soma_dist = np.sum(dist)
        probs = dist / soma_dist if (soma_dist > 0 and not np.isnan(soma_dist)) else np.ones(len(archive_pos)) / len(archive_pos)
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        probs = np.ones(len(archive_pos)) / len(archive_pos)
    return archive_pos[np.random.choice(len(archive_pos), p=probs)]

def run_mohaf_general(problem_name, max_iter=100, num_pop=30, max_archive=50):
    prob = get_problem(problem_name)
    D = prob.n_var
    lb, ub = prob.xl, prob.xu
    
    X = np.random.uniform(lb, ub, size=(num_pop, D))
    X_costs = np.array([prob.evaluate(ind) for ind in X])
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
            if len(archive_pos) > 0:
                best_pos = selecionar_lider_roleta(archive_pos, np.array(archive_costs))
            else:
                best_pos = X[np.random.randint(0, num_pop)]
                
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A, C = 2.0 * a * r1 - a, 2.0 * r2 * np.cos(np.pi * (t / max_iter))
            X_vizinho = X[np.random.randint(0, num_pop)]
            
            X[i] = X_vizinho - A * np.abs(C * X_vizinho - X[i]) if np.abs(A[0]) > 1.0 else best_pos - A * np.abs(C * best_pos - X[i])
            X[i] = np.clip(X[i], lb, ub)
            X_costs[i] = prob.evaluate(X[i])
            
    return np.array(archive_costs)

dtlz_benchmarks = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5"] 
fig = plt.figure(figsize=(18, 11))

print("Gerando Plots Comparativos das Fronteiras 3D (Completo)...")
for idx, b in enumerate(dtlz_benchmarks):
    prob = get_problem(b)
    ref_dirs = get_reference_directions("das-dennis", prob.n_obj, n_partitions=12)
    
    mohaf_pf = run_mohaf_general(b, max_iter=500, num_pop=100)
    res_nsga3 = minimize(prob, NSGA3(ref_dirs=ref_dirs), ('n_gen', 500), verbose=False).F
    res_moead = minimize(prob, MOEAD(ref_dirs=ref_dirs, n_neighbors=15), ('n_gen', 500), verbose=False).F
    
    # Organiza em grade 2x3
    ax = fig.add_subplot(2, 3, idx+1, projection='3d')
    
    if len(mohaf_pf) > 0: 
        ax.scatter(mohaf_pf[:, 0], mohaf_pf[:, 1], mohaf_pf[:, 2], color='blue', s=20, label='MOHAF', alpha=0.7)
    ax.scatter(res_nsga3[:, 0], res_nsga3[:, 1], res_nsga3[:, 2], color='orange', s=12, label='NSGA-III', alpha=0.4, marker='x')
    ax.scatter(res_moead[:, 0], res_moead[:, 1], res_moead[:, 2], color='green', s=12, label='MOEA/D', alpha=0.4, marker='^')
    
    # --- TRATAMENTO DE ESCALA PARA OS HIPER-MULTIMODAIS DTLZ1 e DTLZ3 ---
    if b == "dtlz1":
        ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6); ax.set_zlim(0, 0.6)
    elif b == "dtlz3":
        ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.2); ax.set_zlim(0, 1.2)
        
    ax.set_title(f"Benchmark: {b.upper()}", fontsize=12, fontweight='bold')
    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_zlabel("$f_3$")
    ax.view_init(elev=25, azim=45)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()