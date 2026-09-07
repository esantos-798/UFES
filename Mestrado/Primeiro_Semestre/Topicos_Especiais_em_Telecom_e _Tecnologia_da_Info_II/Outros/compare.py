import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# DEFINIÇÃO DAS FUNÇÕES BENCHMARK (Dimensão D padrão = 10)
# =============================================================================
def sphere(X):
    return np.sum(X**2)

def rastrigin(X):
    return np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10)

def rosenbrock(X):
    return np.sum(100 * (X[1:] - X[:-1]**2)**2 + (1 - X[:-1])**2)

# Configurações do experimento
D = 10            # Número de variáveis (Dimensões da função)
num_pop = 30      # Tamanho da população (indivíduos/partículas/lobos/baleias)
max_iter = 100    # Número máximo de iterações
bounds = [-5.12, 5.12] # Limites do espaço de busca (padrão para Rastrigin/Sphere)

# =============================================================================
# 1. ALGORITMO: HAF (Hydrodynamic Anastomosis Flow) - SUA META-HEURÍSTICA
# =============================================================================
def run_haf(obj_func):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    V = np.zeros((num_pop, D))
    best_pos = np.zeros(D)
    best_score = float('inf')
    history = []
    
    viscosidade = 0.5
    succao = 0.4
    capilaridade = 0.05

    for t in range(max_iter):
        for i in range(num_pop):
            score = obj_func(X[i])
            if score < best_score:
                best_score = score
                best_pos = np.copy(X[i])
        history.append(best_score)
        
        for i in range(num_pop):
            grad_pressao = best_pos - X[i]
            V[i] = viscosidade * V[i] + succao * grad_pressao * np.random.rand(D)
            X[i] = X[i] + V[i]
            
            # Efeito capilar (mutação/ruptura por pressão)
            if np.random.rand() < 0.15:
                X[i] += np.random.normal(0, capilaridade * (bounds[1] - bounds[0]), size=D)
                
            X[i] = np.clip(X[i], bounds[0], bounds[1])
    return best_score, history

# =============================================================================
# 2. ALGORITMO: GWO (Grey Wolf Optimizer)
# =============================================================================
def run_gwo(obj_func):
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
            # Alfa
            A1, C1 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - X[i])
            # Beta
            A2, C2 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - X[i])
            # Delta
            A3, C3 = 2 * a * np.random.rand() - a, 2 * np.random.rand()
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - X[i])
            
            X[i] = (X1 + X2 + X3) / 3
            X[i] = np.clip(X[i], bounds[0], bounds[1])
    return alpha_score, history

# =============================================================================
# 3. ALGORITMO: WOA (Whale Optimization Algorithm)
# =============================================================================
def run_woa(obj_func):
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
# EXECUÇÃO E COMPARAÇÃO GRÁFICA
# =============================================================================
benchmarks = {"Sphere": sphere, "Rastrigin": rastrigin, "Rosenbrock": rosenbrock}

plt.figure(figsize=(15, 4.5))

for idx, (name, func) in enumerate(benchmarks.items(), 1):
    # Executa cada algoritmo
    haf_score, haf_hist = run_haf(func)
    gwo_score, gwo_hist = run_gwo(func)
    woa_score, woa_hist = run_woa(func)
    
    print(f"\n[{name} Function Result - Global Optimum Goal = 0]")
    print(f" > HAF (Seu): {haf_score:.2e}")
    print(f" > GWO:       {gwo_score:.2e}")
    print(f" > WOA:       {woa_score:.2e}")
    
    # Plota gráfico de convergência
    plt.subplot(1, 3, idx)
    plt.plot(haf_hist, label='HAF (Fluidos)', color='blue', linewidth=2)
    plt.plot(gwo_hist, label='GWO (Lobos)', color='orange', linestyle='--')
    plt.plot(woa_hist, label='WOA (Baleias)', color='green', linestyle='-.')
    plt.title(f'Convergência: {name}')
    plt.xlabel('Iterações')
    plt.ylabel('Melhor Fitness (Escala Log)')
    plt.yscale('log') # Escala logarítmica ajuda a ver quem chega mais perto de zero
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()