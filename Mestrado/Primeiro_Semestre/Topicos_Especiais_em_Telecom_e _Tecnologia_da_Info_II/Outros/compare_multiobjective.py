import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.igd import IGD

# =============================================================================
# 1. ADAPTAÇÃO DO SEU MOTOR (MOHAF) PARA O PROVADOR GERAL DA PYMOO
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
    
    # 1. Se todas as distâncias forem infinitas (acontece com poucos pontos ou frentes idênticas)
    if np.all(np.isinf(dist)):
        probs = np.ones(len(archive_pos)) / len(archive_pos)
    else:
        # 2. Trata os pontos de fronteira (infinitos) substituindo pelo dobro do maior valor real encontrado
        valores_reais = dist[~np.isinf(dist)]
        max_real = np.max(valores_reais) if len(valores_reais) > 0 else 1.0
        dist[np.isinf(dist)] = max_real * 2.0
        
        # 3. Garante que nenhuma distância seja negativa ou puramente zero antes da soma
        dist = np.maximum(dist, 1e-6)
        
        # 4. Calcula as probabilidades de forma segura
        soma_dist = np.sum(dist)
        if soma_dist == 0 or np.isnan(soma_dist):
            probs = np.ones(len(archive_pos)) / len(archive_pos)
        else:
            probs = dist / soma_dist
            
    # 5. Validação final anti-NaN redundante antes do sorteio
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

# =============================================================================
# 2. LOOP EXPERIMENTAL DE ALTA PERFORMANCE (10 BENCHMARKS x 10 EXECUÇÕES)
# =============================================================================
benchmarks = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5"]
num_runs = 10
resultados = {b: {"MOHAF": [], "NSGA-III": [], "MOEA/D": []} for b in benchmarks}

print("Iniciando o Torneio Estatístico de Meta-heurísticas...")

for b in benchmarks:
    print(f"\nAvaliando o Benchmark: {b.upper()}")
    prob = get_problem(b)
    pf_teorico = prob.pareto_front() # Fronteira de referência real
    metric_igd = IGD(pf_teorico)
    
    # Configurações de direções para os concorrentes com base no número de objetivos do problema
    n_obj = prob.n_obj
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=30 if n_obj==2 else 12)
    
    for run in range(num_runs):
        print(f" -> Execução {run+1}/{num_runs}...", end="\r")
        
        # 1. Executa MOHAF
        mohaf_pf = run_mohaf_general(b)
        resultados[b]["MOHAF"].append(metric_igd.do(mohaf_pf))
        
        # 2. Executa NSGA-III
        res_nsga3 = minimize(prob, NSGA3(ref_dirs=ref_dirs), ('n_gen', 100), verbose=False)
        resultados[b]["NSGA-III"].append(metric_igd.do(res_nsga3.F))
        
        # 3. Executa MOEA/D
        res_moead = minimize(prob, MOEAD(ref_dirs=ref_dirs, n_neighbors=15), ('n_gen', 100), verbose=False)
        resultados[b]["MOEA/D"].append(metric_igd.do(res_moead.F))

# =============================================================================
# 3. COMPILAÇÃO DOS DADOS EM TABELA ACADÊMICA
# =============================================================================
linhas_tabela = []
for b in benchmarks:
    linha = {
        "Benchmark": b.upper(),
        "MOHAF (Média IGD)": np.mean(resultados[b]["MOHAF"]),
        "MOHAF (Desvio Padrão)": np.std(resultados[b]["MOHAF"]),
        "NSGA-III (Média IGD)": np.mean(resultados[b]["NSGA-III"]),
        "NSGA-III (Desvio Padrão)": np.std(resultados[b]["NSGA-III"]),
        "MOEA/D (Média IGD)": np.mean(resultados[b]["MOEA/D"]),
        "MOEA/D (Desvio Padrão)": np.std(resultados[b]["MOEA/D"])
    }
    linhas_tabela.append(linha)

df_consolidado = pd.DataFrame(linhas_tabela)
print("\n\n=== TABELA CONSOLIDADA DE RESULTADOS (MÉDIA IGD) ===")
print(df_consolidado.to_string(index=False))

# Salva em CSV para você abrir direto no Excel ou usar no LaTeX
df_consolidado.to_csv("resultado_consolidado_mohaf.csv", index=False)
print("\nArquivo 'resultado_consolidado_mohaf.csv' gerado com sucesso!")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

# [Defina aqui as suas funções auxiliares do MOHAF: domina, calcular_crowding_distance e selecionar_lider_roleta]
# [Defina aqui a sua função principal adaptada: run_mohaf_general]

benchmarks = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", "dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5"]

print("Gerando os plots comparativos de alta resolução...")

for b in benchmarks:
    prob = get_problem(b)
    n_obj = prob.n_obj
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=30 if n_obj==2 else 12)
    
    # 1. Executa os algoritmos para a captura visual (1 rodada padrão)
    mohaf_pf = run_mohaf_general(b)
    res_nsga3 = minimize(prob, NSGA3(ref_dirs=ref_dirs), ('n_gen', 100), verbose=False)
    res_moead = minimize(prob, MOEAD(ref_dirs=ref_dirs, n_neighbors=15), ('n_gen', 100), verbose=False)
    
    # 2. Configuração do Plot dependendo do número de Objetivos
    fig = plt.figure(figsize=(8, 6))
    
    if n_obj == 2:
        # Plot 2D para a família ZDT
        plt.scatter(mohaf_pf[:, 0], mohaf_pf[:, 1], color='blue', edgecolors='k', s=50, label='MOHAF (Proposto)', zorder=3)
        plt.scatter(res_nsga3.F[:, 0], res_nsga3.F[:, 1], color='orange', marker='s', edgecolors='k', s=40, label='NSGA-III', alpha=0.8)
        plt.scatter(res_moead.F[:, 0], res_moead.F[:, 1], color='green', marker='^', edgecolors='k', s=40, label='MOEA/D', alpha=0.8)
        
        # Tenta plotar a fronteira teórica se disponível de forma simples
        try:
            pf_teorico = prob.pareto_front()
            if pf_teorico is not None and len(pf_teorico) < 1000:
                # Ordena para não quebrar a linha no plot
                idx_sort = np.argsort(pf_teorico[:, 0])
                plt.plot(pf_teorico[idx_sort, 0], pf_teorico[idx_sort, 1], color='black', linestyle='--', alpha=0.5, label='Fronteira Teórica')
        except:
            pass
            
        plt.xlabel('Objetivo 1 ($f_1$)', fontsize=10)
        plt.ylabel('Objetivo 2 ($f_2$)', fontsize=10)
        
    elif n_obj == 3:
        # Plot 3D para a família DTLZ
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mohaf_pf[:, 0], mohaf_pf[:, 1], mohaf_pf[:, 2], color='blue', edgecolors='k', s=40, label='MOHAF (Proposto)', zorder=3)
        ax.scatter(res_nsga3.F[:, 0], res_nsga3.F[:, 1], res_nsga3.F[:, 2], color='orange', marker='s', edgecolors='k', s=30, label='NSGA-III', alpha=0.7)
        ax.scatter(res_moead.F[:, 0], res_moead.F[:, 1], res_moead.F[:, 2], color='green', marker='^', edgecolors='k', s=30, label='MOEA/D', alpha=0.7)
        
        ax.set_xlabel('Objetivo 1 ($f_1$)', fontsize=10)
        ax.set_ylabel('Objetivo 2 ($f_2$)', fontsize=10)
        ax.set_zlabel('Objetivo 3 ($f_3$)', fontsize=10)
        ax.view_init(elev=30, azim=45) # Ângulo padrão de visualização acadêmica
        
    plt.title(f'Fronteira de Pareto Obtida - {b.upper()}', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Salva cada gráfico como uma imagem PNG de alta resolução (300 DPI)
    plt.savefig(f"plot_comparativo_{b}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("\nTodos os 10 gráficos foram salvos na sua pasta de trabalho atual!")