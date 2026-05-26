import numpy as np
import matplotlib.pyplot as plt

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

# =============================================================================
# CONFIGURAÇÕES GLOBAIS E DICIONÁRIOS DE LIMITES (BOUNDS)
# =============================================================================
D = 10            # Número de dimensões
num_pop = 30      # Tamanho da população
max_iter = 100    # Número de iterações
num_runs = 10     # Número de amostras independentes

# Dicionário de limites rigorosamente ordenado
fun_bounds = {
    "Sphere": [-5.12, 5.12], 
    "Rastrigin": [-5.12, 5.12], 
    "Rosenbrock": [-5.12, 5.12],
    "Ackley": [-32.768, 32.768], 
    "Griewank": [-600.0, 600.0], 
    "Schwefel": [-500.0, 500.0],
    "Quartic Noise": [-1.28, 1.28], 
    "Zakharov": [-5.0, 10.0], 
    "Penal": [-50.0, 50.0],
    "Michalewicz": [0.0, np.pi]
}

# Dicionário de mapeamento para o laço de execução
benchmarks = {
    "Sphere": sphere, "Rastrigin": rastrigin, "Rosenbrock": rosenbrock,
    "Ackley": ackley, "Griewank": griewank, "Schwefel": schwefel_226,
    "Quartic Noise": quartic_noise, "Zakharov": zakharov, "Penal": penal,
    "Michalewicz": michalewicz
}

# =============================================================================
# 2. MOTOR: HAF 1.0 ORIGINAL (ESCOAMENTO LAMINAR PURO - DECAIMENTO LINEAR)
# =============================================================================
def run_haf_1_0(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    V = np.zeros((num_pop, D))
    
    # Avaliação inicial
    fitness = np.array([obj_func(ind) for ind in X])
    leader_idx = np.argmin(fitness)
    leader_score = fitness[leader_idx]
    leader_pos = np.copy(X[leader_idx])
    
    history = []
    
    for t in range(max_iter):
        # --- CARACTERÍSTICA DO HAF 1.0: DECAIMENTO ESTRITAMENTE LINEAR ---
        a = 2.0 * (1.0 - (t / max_iter))
        
        for i in range(num_pop):
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            
            X_vizinho = X[np.random.randint(0, num_pop)]
            
            # Dinâmica de Anastomose Hidráulica Original
            if np.abs(A[0]) > 1.0:
                X[i] = X_vizinho - A * np.abs(C * X_vizinho - X[i])
            else:
                X[i] = leader_pos - A * np.abs(C * leader_pos - X[i])
                
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            
            # Atualiza fitness
            fit_atual = obj_func(X[i])
            if fit_atual < fitness[i]:
                fitness[i] = fit_atual
                if fit_atual < leader_score:
                    leader_score = fit_atual
                    leader_pos = np.copy(X[i])
                    
        history.append(leader_score)
    return leader_score, history

# =============================================================================
# 3. COMPETIDORES PADRÃO (GWO E WOA)
# =============================================================================
def run_gwo(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    history = []
    
    alpha_pos, alpha_score = np.zeros(D), np.inf
    beta_pos, beta_score = np.zeros(D), np.inf
    delta_pos, delta_score = np.zeros(D), np.inf
    
    for t in range(max_iter):
        for i in range(num_pop):
            fit = obj_func(X[i])
            if fit < alpha_score:
                alpha_score, alpha_pos = fit, np.copy(X[i])
            elif fit < beta_score:
                beta_score, beta_pos = fit, np.copy(X[i])
            elif fit < delta_score:
                delta_score, delta_pos = fit, np.copy(X[i])
                
        a = 2.0 * (1.0 - t / max_iter)
        for i in range(num_pop):
            # Cômputo Alpha
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A1, C1 = 2.0 * a * r1 - a, 2.0 * r2
            D_alpha = np.abs(C1 * alpha_pos - X[i])
            X1 = alpha_pos - A1 * D_alpha
            
            # Cômputo Beta
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A2, C2 = 2.0 * a * r1 - a, 2.0 * r2
            D_beta = np.abs(C2 * beta_pos - X[i])
            X2 = beta_pos - A2 * D_beta
            
            # Cômputo Delta
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A3, C3 = 2.0 * a * r1 - a, 2.0 * r2
            D_delta = np.abs(C3 * delta_pos - X[i])
            X3 = delta_pos - A3 * D_delta
            
            X[i] = np.clip((X1 + X2 + X3) / 3.0, bounds[0], bounds[1])
        history.append(alpha_score)
    return alpha_score, history

def run_woa(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    fitness = np.array([obj_func(ind) for ind in X])
    leader_pos = np.copy(X[np.argmin(fitness)])
    leader_score = np.min(fitness)
    history = []
    
    for t in range(max_iter):
        a = 2.0 * (1.0 - t / max_iter)
        for i in range(num_pop):
            p = np.random.rand()
            if p < 0.5:
                r = np.random.rand(D)
                A = 2.0 * a * r - a
                C = 2.0 * np.random.rand(D)
                if np.abs(A[0]) < 1.0:
                    D_leader = np.abs(C * leader_pos - X[i])
                    X[i] = leader_pos - A * D_leader
                else:
                    X_rand = X[np.random.randint(0, num_pop)]
                    D_rand = np.abs(C * X_rand - X[i])
                    X[i] = X_rand - A * D_rand
            else:
                l = np.random.uniform(-1, 1)
                X[i] = np.abs(leader_pos - X[i]) * np.exp(0.5 * l) * np.cos(2 * np.pi * l) + leader_pos
                
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            fit = obj_func(X[i])
            if fit < leader_score:
                leader_score, leader_pos = fit, np.copy(X[i])
        history.append(leader_score)
    return leader_score, history

# =============================================================================
# 4. EXECUÇÃO MULTI-RUN E PLOTAGEM DA GRADE COMPLETA (2x5)
# =============================================================================
benchmarks = {
    "Sphere": sphere, "Rastrigin": rastrigin, "Rosenbrock": rosenbrock,
    "Ackley": ackley, "Griewank": griewank, "Schwefel": schwefel_226,
    "Quartic Noise": quartic_noise, "Zakharov": zakharov, "Penal": penal,
    "Michalewicz": michalewicz
}

# Criar a grade visual padronizada de 2 linhas por 5 colunas
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

print("Iniciando bateria estatística padronizada...")

for idx, (name, func) in enumerate(benchmarks.items()):
    bounds = fun_bounds[name]
    
    haf_all_hist, gwo_all_hist, woa_all_hist = [], [], []
    haf_final_scores, gwo_final_scores, woa_final_scores = [], [], []
    
    print(f"\n>>> Executando {num_runs} runs para: {name}")
    for run in range(num_runs):
        # NOTA: Altere o nome da função abaixo ("run_haf_tuned", "run_haf_ultra", etc.) 
        # conforme o motor específico de cada script!
        haf_score, haf_hist = run_haf_1_0(func, bounds) 
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
        shift = 30.0  # Deslocamento constante e seguro
        ax.plot(haf_mean_hist + shift, label='Seu Algoritmo (Média)', color='blue', linewidth=2)
        ax.plot(gwo_mean_hist + shift, label='GWO (Média)', color='orange', linestyle='--')
        ax.plot(woa_mean_hist + shift, label='WOA (Média)', color='green', linestyle='-.')
        ax.set_ylabel('Fitness + 30 (Log)')
    else:
        eps = 1e-15   # Evita colapso do log se atingir zero absoluto
        ax.plot(np.maximum(haf_mean_hist, eps), label='Seu Algoritmo (Média)', color='blue', linewidth=2)
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
# Altere o título principal dinamicamente conforme a versão rodando
plt.suptitle("Comparativo de Convergência Iterativa Média (10 Runs)", fontsize=16, fontweight='bold', y=1.02)

# Altere o nome do ficheiro para não sobrescrever os resultados das outras versões!
plt.savefig("resultado_padronizado.png", dpi=300, bbox_inches='tight') 
plt.show()