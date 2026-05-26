import numpy as np
import matplotlib.pyplot as plt

class HAFOptimizer:
    def __init__(self, num_links, num_particles=25, max_iter=40):
        self.num_links = num_links
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        # Capacidades físicas reais das interfaces (em Gbps)
        self.link_capacities = np.array([10, 10, 40, 40, 100, 100, 10, 40])
        
        # Matriz de tráfego base que tenta passar pelos links (em Gbps)
        self.link_traffic_base = np.array([8, 9, 35, 12, 85, 40, 7, 38])
        
        # Inicialização das partículas (Cada uma armazena um vetor de pesos IGP para os links)
        # Valores iniciais aleatórios de métricas entre 1 e 100
        self.X = np.random.randint(1, 100, size=(num_particles, num_links)).astype(float)
        
        # Vetor de velocidades das partículas fluidas (fluxo inicialmente estagnado)
        self.V = np.zeros((num_particles, num_links)) 
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []

    def _evaluate_fitness(self, particle_weights):
        """
        Mapeamento Hidrodinâmico: Pesos altos geram atrito (repelem tráfego).
        Pesos baixos reduzem a resistência (atraem fluxo para o link).
        """
        # Normalização e inversão para simular o comportamento de engenharia de tráfego IP
        normalized_weights = particle_weights / np.sum(particle_weights)
        traffic_factor = (1.0 / (normalized_weights + 0.01)) 
        traffic_factor = traffic_factor / np.sum(traffic_factor) * np.sum(self.link_traffic_base)
        
        # O tráfego final no link é influenciado diretamente pela métrica que o engenheiro definiu
        simulated_traffic = 0.5 * self.link_traffic_base + 0.5 * traffic_factor
        
        # Taxa de Utilização do link (U = Banda Consumida / Banda Total)
        U = simulated_traffic / self.link_capacities
        
        # Barreira de Pressão Crítica: Se o link estourar 99% de uso, a perda de pacotes tende ao infinito
        if np.any(U >= 0.99):
            U[U >= 0.99] = 0.99
            
        # Função de Custo Baseada na Teoria das Filas M/M/1 (Queda de Pressão Total do Sistema)
        # Minimizar isso significa eliminar o atraso e evitar o "Tail Drop" nos buffers L2/L3
        cost = np.sum(U / (1.0 - U))
        return cost

    def optimize(self):
        for it in range(self.max_iter):
            costs = np.zeros(self.num_particles)
            
            for i in range(self.num_particles):
                costs[i] = self._evaluate_fitness(self.X[i])
                
                # Registra o "Canal de Menor Resistência" (Melhor Solução Global até aqui)
                if costs[i] < self.best_cost:
                    self.best_cost = costs[i]
                    self.best_solution = np.copy(self.X[i])
            
            self.cost_history.append(self.best_cost)
            
            # --- Dinâmica de Fluidos (Ajuste de Movimento Navier-Stokes) ---
            viscosidade = 0.6   # Inércia: tendência da partícula em manter o fluxo anterior
            succao = 0.4        # Força do gradiente direcionando para a zona de menor pressão
            capilaridade = 0.1  # Coeficiente de expansão/mutação para evitar mínimos locais
            
            for i in range(self.num_particles):
                # Calcula a diferença de pressão em direção à melhor configuração
                gradiente_pressao = self.best_solution - self.X[i]
                
                # Atualização hidrodinâmica da velocidade da partícula
                self.V[i] = (viscosidade * self.V[i] + 
                             succao * gradiente_pressao * np.random.rand(self.num_links))
                
                # A partícula flui para uma nova posição (novo arranjo de métricas IGP)
                self.X[i] = self.X[i] + self.V[i]
                
                # Efeito Anastomótico: Ruptura por pressão (Pequena perturbação estocástica)
                if np.random.rand() < 0.15:
                    self.X[i] += np.random.normal(0, capilaridade * 10, size=self.num_links)
                
                # Garante que as métricas fiquem em um intervalo prático e válido para Roteadores
                self.X[i] = np.clip(self.X[i], 1, 200)
                
        return self.best_solution, self.best_cost, self.cost_history

# Instancia o otimizador para a nossa topologia de 8 links principais no Core
optimizer = HAFOptimizer(num_links=8, num_particles=25, max_iter=40)
best_weights, min_cost, history = optimizer.optimize()

print("-" * 60)
print(f"MÉTRICAS ÓTIMAS ENCONTRADAS PARA OS LINKS OSPF:\n{np.round(best_weights, 1)}")
print(f"QUEDA DE PRESSÃO MÍNIMA (ATRASO MINIMIZADO): {min_cost:.4f}")
print("-" * 60)