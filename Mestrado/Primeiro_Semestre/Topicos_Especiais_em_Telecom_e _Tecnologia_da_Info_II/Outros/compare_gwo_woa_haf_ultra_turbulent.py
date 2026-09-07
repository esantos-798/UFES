import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAÇÕES GLOBAIS DO EXPERIMENTO
# =============================================================================
D = 30            # Configurado para 30 dimensões conforme seu planejamento
num_pop = 30      # Tamanho da população
max_iter = 100    # Número de iterações
num_runs = 10     # Número de amostras independentes

# =============================================================================
# DEFINIÇÃO PADRONIZADA DAS FUNÇÕES BENCHMARK
# =============================================================================
def sphere(x): 
    return np.sum(x**2)

def rastrigin(x): 
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def rosenbrock(x): 
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

def ackley(x):
    d = len(x)
    return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20.0 + np.e

def griewank(x): 
    return np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1.0

def schwefel_226(x): 
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def quartic_noise(x):
    return np.sum(np.arange(1, len(x) + 1) * (x**4)) + np.random.uniform(0, 1)

def alpine_1(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def step_func(x):
    return np.sum(np.floor(x + 0.5)**2)

def michalewicz(x):
    m = 10
    return -np.sum(np.sin(x) * np.sin(np.arange(1, len(x) + 1) * x**2 / np.pi)**(2 * m))

# Dicionário mestre contendo a função e o limite unidimensional base
benchmarks = {
    "Sphere": (sphere, [-5.12, 5.12]),
    "Rastrigin": (rastrigin, [-5.12, 5.12]),
    "Rosenbrock": (rosenbrock, [-2.048, 2.048]),
    "Ackley": (ackley, [-32.768, 32.768]),
    "Griewank": (griewank, [-600.0, 600.0]),
    "Schwefel 2.26": (schwefel_226, [-500.0, 500.0]),
    "Quartic Noise": (quartic_noise, [-1.28, 1.28]),
    "Alpine No.1": (alpine_1, [0.0, 10.0]),
    "Step Function": (step_func, [-100.0, 100.0]),
    "Michalewicz": (michalewicz, [0.0, np.pi])
}

# =============================================================================
# MOTORES DOS ALGORITMOS (HAF TURBO vs COMPEDIDORES)
# =============================================================================
def run_haf_ultra_turbulent(objective_func, bounds, D, max_iter, num_pop, p_turbulento=0.12):
    lb, ub = np.array(bounds[0]), np.array(bounds[1])
    X = np.random.uniform(lb, ub, size=(num_pop, D))
    fit = np.array([objective_func(ind) for ind in X])
    
    gBest_idx = np.argmin(fit)
    gBest = np.copy(X[gBest_idx])
    gBest_fit = fit[gBest_idx]
    
    history = []
    
    for t in range(max_iter):
        a = 2.0 * np.log(1.0 + (max_iter - t) / max_iter) / np.log(2.0)
        
        for i in range(num_pop):
            if np.random.rand() < p_turbulento:
                escala_vortice = 0.01 * (ub - lb) * (1.0 - (t / max_iter))
                X[i] = X[i] + np.random.standard_cauchy(D) * escala_vortice
            else:
                r1, r2 = np.random.rand(D), np.random.rand(D)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2 * np.cos(np.pi * (t / max_iter))
                X_vizinho = X[np.random.randint(0, num_pop)]
                
                if np.abs(A[0]) > 1.0:
                    X[i] = X_vizinho - A * np.abs(C * X_vizinho - X[i])
                else:
                    X[i] = gBest - A * np.abs(C * gBest - X[i])
            
            X[i] = np.clip(X[i], lb, ub)
            fit[i] = objective_func(X[i])
            
            if fit[i] < gBest_fit:
                gBest_fit = fit[i]
                gBest = np.copy(X[i])
                
        history.append(gBest_fit)
                
    return gBest_fit, history

def run_gwo(obj_func, bounds, D, max_iter, num_pop):
    lb, ub = np.array(bounds[0]), np.array(bounds[1])
    X = np.random.uniform(lb, ub, size=(num_pop, D))
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
            X[i] = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
            
    return alpha_score, history

def run_woa(obj_func, bounds, D, max_iter, num_pop):
    lb, ub = np.array(bounds[0]), np.array(bounds[1])
    X = np.random.uniform(lb, ub, size=(num_pop, D))
    best_pos, best_score, history = np.zeros(D), float('inf'), []
    
    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < best_score: 
                best_score = score
                best_pos = np.copy(X[i])
                
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
            X[i] = np.clip(X[i], lb, ub)
            
    return best_score, history

# =============================================================================
# MARATONA QUANTITATIVA E EXTRAPOLAÇÃO GRÁFICA (2x5)
# =============================================================================
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

print(f"Iniciando bateria estatística padronizada (Dimensão D = {D})...")

for idx, (name, (func, limit_range)) in enumerate(benchmarks.items()):
    haf_all, gwo_all, woa_all = [], [], []
    haf_scores, gwo_scores, woa_scores = [], [], []
    
    # Monta os limites vetoriais dinamicamente com base na dimensão D definida
    extended_bounds = [[limit_range[0]] * D, [limit_range[1]] * D]
    
    print(f"\n>>> Executando {num_runs} runs estatísticos para: {name}")
    for r in range(num_runs):
        h_s, h_h = run_haf_ultra_turbulent(func, extended_bounds, D, max_iter, num_pop)
        g_s, g_h = run_gwo(func, extended_bounds, D, max_iter, num_pop)
        w_s, w_h = run_woa(func, extended_bounds, D, max_iter, num_pop)
        
        haf_all.append(h_h); haf_scores.append(h_s)
        gwo_all.append(g_h); gwo_scores.append(g_s)
        woa_all.append(w_h); woa_scores.append(w_s)
    
    print(f" Consolidated [{name}] -> HAF Turbo: {np.mean(haf_scores):.2e} | GWO: {np.mean(gwo_scores):.2e} | WOA: {np.mean(woa_scores):.2e}")
    
    # Médias de convergência iterativa
    mean_haf = np.mean(haf_all, axis=0)
    mean_gwo = np.mean(gwo_all, axis=0)
    mean_woa = np.mean(woa_all, axis=0)
    
    ax = axes[idx]
    
    # Ajuste para valores negativos ou nulos no gráfico logarítmico
    if name == "Michalewicz":
        shift = 30.0  # Deslocamento seguro para manter a curva positiva no log
        ax.plot(mean_haf + shift, label='HAF Ultra Turbo', color='blue', linewidth=2)
        ax.plot(mean_gwo + shift, label='GWO', color='orange', linestyle='--')
        ax.plot(mean_woa + shift, label='WOA', color='green', linestyle='-.')
        ax.set_ylabel('Fitness + 30 (Log)')
    else:
        eps = 1e-15   # Evita colapso do logaritmo se atingir zero absoluto
        ax.plot(np.maximum(mean_haf, eps), label='HAF Ultra Turbo', color='blue', linewidth=2)
        ax.plot(np.maximum(mean_gwo, eps), label='GWO', color='orange', linestyle='--')
        ax.plot(np.maximum(mean_woa, eps), label='WOA', color='green', linestyle='-.')
        if idx % 5 == 0: 
            ax.set_ylabel('Melhor Fitness Médio (Log)')

    ax.set_yscale('log')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Iterações', fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    
    if idx == 0: 
        ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.suptitle(f"Comparativo de Convergência Iterativa Média (D={D} | {num_runs} Runs) - HAF Ultra Turbulent", fontsize=16, fontweight='bold', y=1.02)

# Salva o arquivo especificando o motor turbulento para não gerar conflito de arquivos
plt.savefig("resultado_haf_ultra_turbulent.png", dpi=300, bbox_inches='tight') 
plt.show()