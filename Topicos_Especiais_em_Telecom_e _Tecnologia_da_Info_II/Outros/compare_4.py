import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. DEFINIÇÃO DOS BENCHMARKS (OS SEIS MAIS FAMOSOS DA LITERATURA)
# =============================================================================
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

def ackley(x):
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + 20.0 + np.e

def griewank(x):
    sum_part = np.sum(x**2) / 4000.0
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1.0

def schwefel_226(x):
    # O mínimo global desta função ocorre em x = 420.9687 para todas as dimensões, resultando em 0
    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# =============================================================================
# 2. ALGORITMOS DE OTIMIZAÇÃO (CONFIGURADOS PARA O AMBIENTE DE TESTES)
# =============================================================================
D = 10
num_pop = 30
max_iter = 100

def run_haf_ultra(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    best_pos = np.zeros(D)
    best_score = float('inf')
    history = []
    
    cont_estagnacao = 0
    last_best = float('inf')
    
    # Atritores caóticos de Henon para turbulência avançada
    hx, hy = 0.1, 0.3

    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < best_score:
                best_score = score
                best_pos = np.copy(X[i])
        history.append(best_score)
        
        if best_score == last_best:
            cont_estagnacao += 1
        else:
            cont_estagnacao = 0
            last_best = best_score

        a = 2.0 - t * (2.0 / max_iter)
        
        for i in range(num_pop):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2 * np.cos(np.pi * (t / max_iter))
            
            X_vizinho = X[np.random.randint(0, num_pop)]
            
            if np.abs(A[0]) > 1.0:
                D_fluxo = np.abs(C * X_vizinho - X[i])
                X[i] = X_vizinho - A * D_fluxo
            else:
                D_ralo = np.abs(C * best_pos - X[i])
                X[i] = best_pos - A * D_ralo
            
            # --- MELHORIA ULTRA: TURBULÊNCIA POR MAPA CAÓTICO DE HENON ---
            if cont_estagnacao > 3:
                if np.random.rand() < 0.3: # Aumentamos levemente a sensibilidade
                    # Equações do Mapa Caótico de Henon para gerar varredura não-linear pura
                    hx_new = 1 - 1.4 * hx**2 + hy
                    hy = 0.3 * hx
                    hx = hx_new
                    # Injeta uma perturbação pseudo-aleatória determinística escalonada no tempo
                    fator_escala = 0.05 * (1.0 - (t / max_iter))
                    X[i] = best_pos + hx * fator_escala * (bounds[1] - bounds[0])
            
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            
        if cont_estagnacao > 3:
            cont_estagnacao = 0
            
    return best_score, history

# --- IMPLEMENTAÇÃO PADRÃO SIMPLIFICADA DO GWO ---
def run_gwo(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    alpha_pos, beta_pos, delta_pos = np.zeros(D), np.zeros(D), np.zeros(D)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    history = []
    
    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < alpha_score:
                delta_score = beta_score; delta_pos = np.copy(beta_pos)
                beta_score = alpha_score; beta_pos = np.copy(alpha_pos)
                alpha_score = score; alpha_pos = np.copy(X[i])
            elif score < beta_score:
                delta_score = beta_score; delta_pos = np.copy(beta_pos)
                beta_score = score; beta_pos = np.copy(X[i])
            elif score < delta_score:
                delta_score = score; delta_pos = np.copy(X[i])
        history.append(alpha_score)
        
        a = 2.0 - t * (2.0 / max_iter)
        for i in range(num_pop):
            # Lobo Alpha
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A1, C1 = 2.0 * a * r1 - a, 2.0 * r2
            D_alpha = np.abs(C1 * alpha_pos - X[i])
            X1 = alpha_pos - A1 * D_alpha
            # Lobo Beta
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A2, C2 = 2.0 * a * r1 - a, 2.0 * r2
            D_beta = np.abs(C2 * beta_pos - X[i])
            X2 = beta_pos - A2 * D_beta
            # Lobo Delta
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A3, C3 = 2.0 * a * r1 - a, 2.0 * r2
            D_delta = np.abs(C3 * delta_pos - X[i])
            X3 = delta_pos - A3 * D_delta
            
            X[i] = (X1 + X2 + X3) / 3.0
            X[i] = np.clip(X[i], bounds[0], bounds[1])
    return alpha_score, history

# --- IMPLEMENTAÇÃO PADRÃO SIMPLIFICADA DO WOA ---
def run_woa(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    best_pos = np.zeros(D)
    best_score = float('inf')
    history = []
    
    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < best_score:
                best_score = score; best_pos = np.copy(X[i])
        history.append(best_score)
        
        a = 2.0 - t * (2.0 / max_iter)
        for i in range(num_pop):
            p = np.random.rand()
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            
            if p < 0.5:
                if np.abs(A[0]) < 1.0:
                    D_leader = np.abs(C * best_pos - X[i])
                    X[i] = best_pos - A * D_leader
                else:
                    X_rand = X[np.random.randint(0, num_pop)]
                    D_rand = np.abs(C * X_rand - X[i])
                    X[i] = X_rand - A * D_rand
            else:
                D_leader = np.abs(best_pos - X[i])
                l = np.random.uniform(-1, 1)
                X[i] = D_leader * np.exp(0.5 * l) * np.cos(2.0 * np.pi * l) + best_pos
                
            X[i] = np.clip(X[i], bounds[0], bounds[1])
    return best_score, history

# =============================================================================
# 3. PROCESSO EXPERIMENTAL QUANTITATIVO (10 RUNS POR BENCHMARK)
# =============================================================================
benchmarks = {
    "Sphere": (sphere, [-5.12, 5.12]),
    "Rastrigin": (rastrigin, [-5.12, 5.12]),
    "Rosenbrock": (rosenbrock, [-2.048, 2.048]),
    "Ackley": (ackley, [-32.768, 32.768]),
    "Griewank": (griewank, [-600.0, 600.0]),
    "Schwefel 2.26": (schwefel_226, [-500.0, 500.0])
}

num_runs = 10
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, (func, bounds)) in enumerate(benchmarks.items()):
    haf_all, gwo_all, woa_all = [], [], []
    haf_scores, gwo_scores, woa_scores = [], [], []
    
    print(f"\n>>> Executando {num_runs} Amostras Estatísticas para: {name}")
    
    for r in range(num_runs):
        h_s, h_h = run_haf_ultra(func, bounds)
        g_s, g_h = run_gwo(func, bounds)
        w_s, w_h = run_woa(func, bounds)
        
        haf_all.append(h_h); haf_scores.append(h_s)
        gwo_all.append(g_h); gwo_scores.append(g_s)
        woa_all.append(w_h); woa_scores.append(w_s)
    
    # Médias de convergência iterativa
    haf_mean = np.mean(haf_all, axis=0)
    gwo_mean = np.mean(gwo_all, axis=0)
    woa_mean = np.mean(woa_all, axis=0)
    
    # Log de resultados consolidados no console
    print(f" Concluído! [RESULTADOS FINAIS - {name}]")
    print(f"  * HAF Ultra | Média: {np.mean(haf_scores):.2e} | DP: {np.std(haf_scores):.2e}")
    print(f"  * GWO       | Média: {np.mean(gwo_scores):.2e} | DP: {np.std(gwo_scores):.2e}")
    print(f"  * WOA       | Média: {np.mean(woa_scores):.2e} | DP: {np.std(woa_scores):.2e}")
    
    # Plotagem da grade de gráficos
    ax = axes[idx]
    ax.plot(haf_mean, label='HAF Ultra (Média)', color='blue', linewidth=2)
    ax.plot(gwo_mean, label='GWO (Média)', color='orange', linestyle='--')
    ax.plot(woa_mean, label='WOA (Média)', color='green', linestyle='-.')
    
    ax.set_title(f'Convergência Média: {name}')
    ax.set_xlabel('Iterações')
    ax.set_ylabel('Fitness Médio (Log)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()