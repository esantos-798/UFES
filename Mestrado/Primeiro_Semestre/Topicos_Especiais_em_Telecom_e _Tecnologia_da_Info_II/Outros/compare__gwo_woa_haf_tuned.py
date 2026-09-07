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
# 1. EVOLUÇÃO MATEMÁTICA: MOTOR HAF TUNED (ESCOAMENTO CALIBRADO E ESTÁVEL)
# =============================================================================
def run_haf_tuned(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    best_pos = np.zeros(D)
    best_score = float('inf')
    history = []
    
    cont_estagnacao = 0
    last_best = float('inf')

    for t in range(max_iter):
        # Elitismo / Mapeamento do ponto de menor pressão
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < best_score:
                best_score = score
                best_pos = np.copy(X[i])
        history.append(best_score)
        
        # Contador de estagnação para ativar a quebra de mínimos locais
        if best_score == last_best:
            cont_estagnacao += 1
        else:
            cont_estagnacao = 0
            last_best = best_score

        # Coeficiente de escoamento que decai linearmente de 2 para 0
        a = 2.0 - t * (2.0 / max_iter)
        
        for i in range(num_pop):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            
            # Coeficiente A padrão (Exploração vs Explotação)
            A = 2.0 * a * r1 - a
            
            # --- CALIBRAÇÃO TUNED: COEFICIENTE C WAVELET DINÂMICO ---
            # O cosseno força uma oscilação na pressão do fluido para evitar travamento prematuro
            C = 2.0 * r2 * np.cos(np.pi * (t / max_iter))
            
            X_vizinho = X[np.random.randint(0, num_pop)]
            
            # Dinâmica de Anastomose Hidráulica Calibrada
            if np.abs(A[0]) > 1.0:
                D_fluxo = np.abs(C * X_vizinho - X[i])
                X[i] = X_vizinho - A * D_fluxo
            else:
                D_ralo = np.abs(C * best_pos - X[i])
                X[i] = best_pos - A * D_ralo
            
            # --- TURBULÊNCIAS LOCAIS: Mecanismo anticoágulo ---
            if cont_estagnacao > 3:
                if np.random.rand() < 0.2:
                    # Injeção capilar ultra-focada ao redor do líder atual para sacudir a população
                    X[i] = best_pos + np.random.normal(0, 0.01, size=D)
            
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            
        if cont_estagnacao > 3:
            cont_estagnacao = 0
            
    return best_score, history

# =============================================================================
# 2. ALGORITMO COMPLEMENTAR: GWO (Grey Wolf Optimizer)
# =============================================================================
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
        
        a = 2 - t * (2 / max_iter)
        for i in range(num_pop):
            A1, C1 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - X[i])
            
            A2, C2 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - X[i])
            
            A3, C3 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - X[i])
            
            X[i] = (X1 + X2 + X3) / 3
            X[i] = np.clip(X[i], bounds[0], bounds[1])
    return alpha_score, history

# =============================================================================
# 3. ALGORITMO COMPLEMENTAR: WOA (Whale Optimization Algorithm)
# =============================================================================
def run_woa(obj_func, bounds):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    leader_pos = np.zeros(D)
    leader_score = float('inf')
    history = []
    b = 1

    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < leader_score:
                leader_score = score
                leader_pos = np.copy(X[i])
        history.append(leader_score)
        
        a = 2 - t * (2 / max_iter)
        for i in range(num_pop):
            A = 2 * a * np.random.rand() - a
            C = 2 * np.random.rand()
            p = np.random.rand()
            l = np.random.uniform(-1, 1)
            
            if p < 0.5:
                if np.abs(A) < 1:
                    X[i] = leader_pos - A * np.abs(C * leader_pos - X[i])
                else:
                    X_rand = X[np.random.randint(0, num_pop)]
                    X[i] = X_rand - A * np.abs(C * X_rand - X[i])
            else:
                X[i] = np.abs(leader_pos - X[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos
                
            X[i] = np.clip(X[i], bounds[0], bounds[1])
    return leader_score, history

# =============================================================================
# 4. EXECUÇÃO EXPERIMENTAL CONSOLIDADA: GRADE DE 10 BENCHMARKS (2x5)
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
        haf_score, haf_hist = run_haf_tuned(func, bounds) 
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