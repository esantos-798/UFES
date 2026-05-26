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

# Configurações globais do experimento
D = 10            # Número de dimensões
num_pop = 30      # Tamanho da população
max_iter = 100    # Número de iterações
bounds = [-5.12, 5.12] 

# =============================================================================
# 1. HAF TUNED (Hydrodynamic Anastomosis Flow v2)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# FUNÇÕES BENCHMARK
# =============================================================================
def sphere(X): return np.sum(X**2)
def rastrigin(X): return np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10)
def rosenbrock(X): return np.sum(100 * (X[1:] - X[:-1]**2)**2 + (1 - X[:-1])**2)

D = 10            
num_pop = 30      
max_iter = 100    
bounds = [-5.12, 5.12] 

# =============================================================================
# HAF TUNADO (Hydrodynamic Anastomosis Flow v2)
# =============================================================================
def run_haf_tuned(obj_func):
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    best_pos = np.zeros(D)
    best_score = float('inf')
    history = []
    
    cont_estagnacao = 0
    last_best = float('inf')

    for t in range(max_iter):
        # 1. ELITISMO: Mapeia o ponto de menor pressão
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

        # 2. DINÂMICA DE FLUXO POR ENVELOPAMENTO (Sem Vetor V)
        # Coeficiente de escoamento que decai linearmente de 2 para 0 (Exploração -> Explotação)
        a = 2.0 - t * (2.0 / max_iter)
        
        for i in range(num_pop):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            
            # Coeficiente A padrão (comandando exploração vs explotação)
            A = 2.0 * a * r1 - a
            
            # --- CALIBRAÇÃO NOVA: COEFICIENTE C WAVELET DINÂMICO ---
            # Em vez de puramente aleatório, C agora oscila simulando ondas de pressão.
            # Isso impede que o fluido estagne nos cossenos da Rastrigin.
            C = 2.0 * r2 * np.cos(np.pi * (t / max_iter))
            
            # Seleciona outro canal de fluxo (outra partícula) aleatoriamente
            X_vizinho = X[np.random.randint(0, num_pop)]
            
            if np.abs(A[0]) > 1.0:
                D_fluxo = np.abs(C * X_vizinho - X[i])
                X[i] = X_vizinho - A * D_fluxo
            else:
                D_ralo = np.abs(C * best_pos - X[i])
                X[i] = best_pos - A * D_ralo
            
            # 3. TURBULÊNCIAS LOCAIS (Se o ralo entupir / estagnar)
            if cont_estagnacao > 3:
                if np.random.rand() < 0.2:
                    # Injeção capilar ultra-focada para quebrar o mínimo local da Rastrigin
                    X[i] = best_pos + np.random.normal(0, 0.01, size=D)
            
            # Controle térmico de fronteira (Clip tradicional para garantir estabilidade)
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            
        if cont_estagnacao > 3:
            cont_estagnacao = 0
            
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
# EXECUÇÃO E COMPARAÇÃO GRÁFICA CORRIGIDA
# =============================================================================
benchmarks = {"Sphere": sphere, "Rastrigin": rastrigin, "Rosenbrock": rosenbrock}

plt.figure(figsize=(15, 4.5))

for idx, (name, func) in enumerate(benchmarks.items(), 1):
    # Executa cada algoritmo chamando os nomes corretos
    haf_score, haf_hist = run_haf_tuned(func)
    gwo_score, gwo_hist = run_gwo(func)
    woa_score, woa_hist = run_woa(func)
    
    print(f"\n[{name} - Ótimo Global Alvo = 0]")
    print(f" > HAF Tuned (Seu): {haf_score:.2e}")
    print(f" > GWO:             {gwo_score:.2e}")
    print(f" > WOA:             {woa_score:.2e}")
    
    # Geração dos subplots
    plt.subplot(1, 3, idx)
    plt.plot(haf_hist, label='HAF Tuned (Fluidos)', color='blue', linewidth=2)
    plt.plot(gwo_hist, label='GWO (Lobos)', color='orange', linestyle='--')
    plt.plot(woa_hist, label='WOA (Baleias)', color='green', linestyle='-.')
    plt.title(f'Convergência: {name}')
    plt.xlabel('Iterações')
    plt.ylabel('Melhor Fitness (Escala Log)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()