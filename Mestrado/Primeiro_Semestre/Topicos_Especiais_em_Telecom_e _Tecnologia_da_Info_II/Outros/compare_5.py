import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. DEFINIÇÃO DOS 10 BENCHMARKS DA LITERATURA
# =============================================================================
def sphere(x): return np.sum(x**2)
def rastrigin(x): return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)
def rosenbrock(x): return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)
def ackley(x):
    d = len(x)
    return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20.0 + np.e
def griewank(x): return np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1.0
def schwefel_226(x): return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# --- NOVOS BENCHMARKS ---
def quartic_noise(x):
    # Adiciona ruído aleatório uniforme na avaliação da bacia x^4
    return np.sum(np.arange(1, len(x) + 1) * (x**4)) + np.random.rand()

def alpine_1(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))

def step_func(x):
    return np.sum(np.floor(x + 0.5)**2)

def michalewicz(x, m=10):
    i = np.arange(1, len(x) + 1)
    return -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi))**(2 * m))

# =============================================================================
# 2. CONFIGURAÇÃO DE AMBIENTE E ALGORITMOS
# =============================================================================
D = 10
num_pop = 30
max_iter = 100

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
        #a = 2.0 - t * (2.0 / max_iter)
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
            
            if cont_estagnacao > 3 and np.random.rand() < 0.3:
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
# 3. MARATONA QUANTITATIVA DE 10 BENCHMARKS (10 RUNS INDEPENDENTES)
# =============================================================================
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

num_runs = 10
fig, axes = plt.subplots(2, 5, figsize=(22, 9)) # Grade 2x5 limpa
axes = axes.flatten()

for idx, (name, (func, bounds)) in enumerate(benchmarks.items()):
    haf_all, gwo_all, woa_all = [], [], []
    haf_scores, gwo_scores, woa_scores = [], [], []
    
    print(f"Executando {num_runs} runs estatísticos para: {name}...")
    for r in range(num_runs):
        h_s, h_h = run_haf_ultra(func, bounds)
        g_s, g_h = run_gwo(func, bounds)
        w_s, w_h = run_woa(func, bounds)
        haf_all.append(h_h); haf_scores.append(h_s)
        gwo_all.append(g_h); gwo_scores.append(g_s)
        woa_all.append(w_h); woa_scores.append(w_s)
    
    print(f" Consolidated [{name}] -> HAF: {np.mean(haf_scores):.2e} | GWO: {np.mean(gwo_scores):.2e} | WOA: {np.mean(woa_scores):.2e}")
    
    # Plotagem da grade de gráficos
    ax = axes[idx]
    
    # --- AJUSTE PARA FUNÇÕES COM VALORES NEGATIVOS (MICHALEWICZ) ---
    if name == "Michalewicz":
        # Desloca os valores para cima baseado no mínimo teórico conhecido para não quebrar o log
        shift = 10.0 
        ax.plot(np.mean(haf_all, axis=0) + shift, label='HAF Ultra', color='blue', linewidth=2)
        ax.plot(np.mean(gwo_all, axis=0) + shift, label='GWO', color='orange', linestyle='--')
        ax.plot(np.mean(woa_all, axis=0) + shift, label='WOA', color='green', linestyle='-.')
        ax.set_ylabel('Fitness + 10 (Log)')
    else:
        # Plotagem padrão para as outras 9 funções
        ax.plot(np.mean(haf_all, axis=0), label='HAF Ultra', color='blue', linewidth=2)
        ax.plot(np.mean(gwo_all, axis=0), label='GWO', color='orange', linestyle='--')
        ax.plot(np.mean(woa_all, axis=0), label='WOA', color='green', linestyle='-.')
        if idx % 5 == 0: ax.set_ylabel('Fitness Médio (Log)')

    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    if idx >= 5: ax.set_xlabel('Iterações')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()