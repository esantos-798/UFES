import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================
D = 10            # Número de dimensões
num_pop = 30      # Tamanho da população
max_iter = 100    # Número de iterações
num_runs = 10     # Número de amostras independentes

# =============================================================================
# DEFINIÇÃO PADRONIZADA DAS 10 FUNÇÕES BENCHMARK
# =============================================================================
def sphere(X):
    return np.sum(X**2)

def rastrigin(X):
    return np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10)

def rosenbrock(X):
    return np.sum(100 * (X[1:] - X[:-1]**2)**2 + (1 - X[:-1])**2)

def ackley(X):
    d = len(X)
    return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(X**2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * X)) / d) + 20.0 + np.e

def griewank(X):
    return np.sum(X**2) / 4000.0 - np.prod(np.cos(X / np.sqrt(np.arange(1, len(X) + 1)))) + 1.0

def schwefel_226(X):
    return 418.9829 * len(X) - np.sum(X * np.sin(np.sqrt(np.abs(X))))

def quartic_noise(X):
    return np.sum(np.arange(1, len(X) + 1) * (X**4)) + np.random.uniform(0, 1)

def zakharov(X):
    sum1 = np.sum(X**2)
    sum2 = np.sum(0.5 * np.arange(1, len(X) + 1) * X)
    return sum1 + sum2**2 + sum2**4

def penal(X):
    def u(xi, a, k, m):
        if xi > a: return k * (xi - a)**m
        elif xi < -a: return k * (-xi - a)**m
        return 0
    d = len(X)
    y = 1.0 + (X + 1.0) / 4.0
    term1 = 10.0 * np.sin(np.pi * y[0])**2
    term2 = np.sum((y[:-1] - 1.0)**2 * (1.0 + 10.0 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1.0)**2
    sum_u = np.sum([u(xi, 10, 100, 4) for xi in X])
    return (np.pi / d) * (term1 + term2 + term3) + sum_u

def michalewicz(X):
    m = 10
    return -np.sum(np.sin(X) * np.sin(np.arange(1, len(X) + 1) * X**2 / np.pi)**(2 * m))

# Dicionários Perfeitamente Sincronizados (Mesmos Nomes e Mesma Ordem)
fun_bounds = {
    "Sphere": [-5.12, 5.12], 
    "Rastrigin": [-5.12, 5.12], 
    "Rosenbrock": [-2.048, 2.048],
    "Ackley": [-32.768, 32.768], 
    "Griewank": [-600.0, 600.0], 
    "Schwefel": [-500.0, 500.0],
    "Quartic Noise": [-1.28, 1.28], 
    "Zakharov": [-5.0, 10.0], 
    "Penal": [-50.0, 50.0],
    "Michalewicz": [0.0, np.pi]
}

benchmarks = {
    "Sphere": sphere, 
    "Rastrigin": rastrigin, 
    "Rosenbrock": rosenbrock,
    "Ackley": ackley, 
    "Griewank": griewank, 
    "Schwefel": schwefel_226,
    "Quartic Noise": quartic_noise, 
    "Zakharov": zakharov, 
    "Penal": penal,
    "Michalewicz": michalewicz
}

# =============================================================================
# 2. MOTORES DOS ALGORITMOS
# =============================================================================
def run_haf_ultra(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    best_pos, best_score = np.zeros(D), float('inf')
    history, cont_estagnacao, last_best = [], 0, float('inf')
    hx, hy = 0.1, 0.3 # Caos de Henon

    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < best_score:
                best_score = score; best_pos = np.copy(X[i])
        history.append(best_score)
        
        cont_estagnacao = cont_estagnacao + 1 if best_score == last_best else 0
        last_best = best_score
        
        a = 2.0 * np.log(1.0 + (max_iter - t) / max_iter) / np.log(2.0)
        
        for i in range(num_pop):
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2 * np.cos(np.pi * (t / max_iter))
            X_vizinho = X[np.random.randint(0, num_pop)]
            
            if np.abs(A[0]) > 1.0:
                X[i] = X_vizinho - A * np.abs(C * X_vizinho - X[i])
            else:
                X[i] = best_pos - A * np.abs(C * best_pos - X[i])
            
            # --- OPERADOR DE EXTRAPOLAÇÃO DIRECIONAL ---
            if cont_estagnacao > 3:
                if np.random.rand() < 0.2:
                    rand_dim = np.random.randint(0, D)
                    X[i][rand_dim] = best_pos[rand_dim] + np.random.uniform(-0.1, 0.1) * (best_pos[rand_dim] - X[i][rand_dim])
                
                elif np.random.rand() < 0.3:
                    hx_new = 1 - 1.4 * hx**2 + hy
                    hy = 0.3 * hx; hx = hx_new
                    X[i] = best_pos + hx * 0.05 * (1.0 - (t / max_iter)) * (bounds[1] - bounds[0])
                    
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            
        if cont_estagnacao > 3: cont_estagnacao = 0
    return best_score, history

def run_gwo(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    alpha_pos, beta_pos, delta_pos = np.zeros(D), np.zeros(D), np.zeros(D)
    alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
    history = []
    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < alpha_score:
                delta_score, delta_pos = beta_score, np.copy(beta_pos)
                beta_score, beta_pos = alpha_score, np.copy(alpha_pos)
                alpha_score, alpha_pos = score, np.copy(X[i])
            elif score < beta_score:
                delta_score, delta_pos = beta_score, np.copy(beta_pos)
                beta_score, beta_pos = score, np.copy(X[i])
            elif score < delta_score:
                delta_score, delta_pos = score, np.copy(X[i])
        history.append(alpha_score)
        a = 2.0 - t * (2.0 / max_iter)
        for i in range(num_pop):
            X1 = alpha_pos - (2.0*a*np.random.rand(D)-a) * np.abs(2.0*np.random.rand(D)*alpha_pos - X[i])
            X2 = beta_pos - (2.0*a*np.random.rand(D)-a) * np.abs(2.0*np.random.rand(D)*beta_pos - X[i])
            X3 = delta_pos - (2.0*a*np.random.rand(D)-a) * np.abs(2.0*np.random.rand(D)*delta_pos - X[i])
            X[i] = np.clip((X1 + X2 + X3) / 3.0, bounds[0], bounds[1])
    return alpha_score, history

def run_woa(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    best_pos, best_score, history = np.zeros(D), float('inf'), []
    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < best_score: best_score = score; best_pos = np.copy(X[i])
        history.append(best_score)
        a = 2.0 - t * (2.0 / max_iter)
        for i in range(num_pop):
            p, A, C = np.random.rand(), (2.0*a*np.random.rand(D)-a), 2.0*np.random.rand(D)
            if p < 0.5:
                if np.abs(A[0]) < 1.0:
                    X[i] = best_pos - A * np.abs(C * best_pos - X[i])
                else:
                    X_rand = X[np.random.randint(0, num_pop)]
                    X[i] = X_rand - A * np.abs(C * X_rand - X[i])
            else:
                X[i] = np.abs(best_pos - X[i]) * np.exp(0.5*np.random.uniform(-1,1)) * np.cos(2.0*np.pi*np.random.uniform(-1,1)) + best_pos
            X[i] = np.clip(X[i], bounds[0], bounds[1])
    return best_score, history

# =============================================================================
# 3. EXECUÇÃO E GENERATION DA GRADE VISUAL 2x5
# =============================================================================
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

print("Iniciando bateria estatística padronizada...")

for idx, (name, func) in enumerate(benchmarks.items()):
    bounds = fun_bounds[name]
    
    haf_all_hist, gwo_all_hist, woa_all_hist = [], [], []
    haf_final_scores, gwo_final_scores, woa_final_scores = [], [], []
    
    print(f"\n>>> Executando {num_runs} runs para: {name}")
    for run in range(num_runs):
        # NOTA: Troque aqui o motor do seu algoritmo para as outras versões!
        haf_score, haf_hist = run_haf_ultra(func, bounds) 
        gwo_score, gwo_hist = run_gwo(func, bounds)
        woa_score, woa_hist = run_woa(func, bounds)
        
        haf_all_hist.append(haf_hist)
        gwo_all_hist.append(gwo_hist)
        woa_all_hist.append(woa_hist)
        
        haf_final_scores.append(haf_score)
        gwo_final_scores.append(gwo_score)
        woa_final_scores.append(woa_score)
        
    # Médias de convergência iterativa
    haf_mean_hist = np.mean(haf_all_hist, axis=0)
    gwo_mean_hist = np.mean(gwo_all_hist, axis=0)
    woa_mean_hist = np.mean(woa_all_hist, axis=0)
    
    ax = axes[idx]
    
    # Tratamento do eixo logarítmico para funções negativas ou nulas
    if name == "Michalewicz":
        shift = 30.0  
        ax.plot(haf_mean_hist + shift, label='HAF Ultra (Média)', color='blue', linewidth=2)
        ax.plot(gwo_mean_hist + shift, label='GWO (Média)', color='orange', linestyle='--')
        ax.plot(woa_mean_hist + shift, label='WOA (Média)', color='green', linestyle='-.')
        ax.set_ylabel('Fitness + 30 (Log)')
    else:
        eps = 1e-15   
        ax.plot(np.maximum(haf_mean_hist, eps), label='HAF Ultra (Média)', color='blue', linewidth=2)
        ax.plot(np.maximum(gwo_mean_hist, eps), label='GWO (Média)', color='orange', linestyle='--')
        ax.plot(np.maximum(woa_mean_hist, eps), label='WOA (Média)', color='green', linestyle='-.')
        ax.set_ylabel('Melhor Fitness Médio (Log)')
        
    ax.set_yscale('log')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Iterações', fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    
    if idx == 0:
        ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.suptitle("Comparativo de Convergência Iterativa Média (10 Runs) - HAF Ultra", fontsize=16, fontweight='bold', y=1.02)

# Troque o nome do arquivo string por script para salvar separado!
plt.savefig("resultado_haf_ultra.png", dpi=300, bbox_inches='tight') 
plt.show()