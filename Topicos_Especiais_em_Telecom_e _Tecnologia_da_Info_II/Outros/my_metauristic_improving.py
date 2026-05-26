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
    # Inicialização do sistema
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    V = np.zeros((num_pop, D))
    best_pos = np.zeros(D)
    best_score = float('inf')
    history = []
    
    cont_estagnacao = 0
    last_best = float('inf')

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
            
        # ---------------------------------------------------------------------
        # FÍSICA NOVA 1: VISCOSIDADE NÃO-NEWTONIANA DINÂMICA
        # O fluido fica "fino" (caos alto) no início e espesso (refinamento) no fim.
        # Adicionamos um decaimento não-linear exponencial para acelerar a descida.
        # ---------------------------------------------------------------------
        fator_tempo = np.exp(-3 * (t / max_iter))
        viscosidade = 0.4 * fator_tempo
        
        # A força de sucção (pressão) cresce de forma inversa e agressiva
        succao = 0.8 * (1.0 - fator_tempo)
        
        # ---------------------------------------------------------------------
        # FÍSICA NOVA 2: VÓRTICES DE TURBULÊNCIA DE KOLMOGOROV (ANTI-ESTAGNAÇÃO)
        # Se travar, em vez de salto aleatório, o fluido rotaciona o vetor de velocidade
        # usando uma distribuição de Levy ou perturbação de escala decrescente.
        # ---------------------------------------------------------------------
        if cont_estagnacao > 2: # Mais sensível (dispara com 3 iterações parado)
            raio_vortice = 0.2 * (1.0 - (t / max_iter))
            for i in range(num_pop):
                # Zera a inércia que causava o redemoinho cego
                V[i] = np.zeros(D)
                # Fragmentação do fluxo: espalha soluções em um raio ultra-focado
                X[i] = best_pos + np.random.laplace(0, raio_vortice, size=D)
            cont_estagnacao = 0

        # ---------------------------------------------------------------------
        # FÍSICA NOVA 3: EQUAÇÃO DE MOVIMENTO TURBULENTO
        # Introduzimos uma componente de cruzamento de informações (Anastomose Real).
        # Uma partícula não olha só para o líder, ela sofre atração de uma partícula aleatória
        # para garantir a troca de coordenadas (cruzamento).
        # ---------------------------------------------------------------------
        for i in range(num_pop):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            
            # Escolhe outra veia/canal de fluxo aleatório para interagir
            X_vizinho = X[np.random.randint(0, num_pop)]
            
            # Gradiente duplo: Pressão para o líder + Gradiente de cisalhamento com o vizinho
            grad_pressao = best_pos - X[i]
            grad_cisalhamento = X_vizinho - X[i]
            
            # Atualização da velocidade relativística
            V[i] = viscosidade * V[i] + succao * r1 * grad_pressao + 0.1 * r2 * grad_cisalhamento
            X[i] = X[i] + V[i]
            
            # ---------------------------------------------------------------------
            # FÍSICA NOVA 4: DISSIPAÇÃO TÉRMICA DE FRONTEIRA
            # Se a partícula colidir com o limite do mapa, a velocidade naquela dimensão 
            # morre instantaneamente (absorção de impacto). Evita o efeito chicote.
            # ---------------------------------------------------------------------
            for d in range(D):
                if X[i, d] < bounds[0]:
                    X[i, d] = bounds[0]
                    V[i, d] = 0.0 
                elif X[i, d] > bounds[1]:
                    X[i, d] = bounds[1]
                    V[i, d] = 0.0
            
    return best_score, history