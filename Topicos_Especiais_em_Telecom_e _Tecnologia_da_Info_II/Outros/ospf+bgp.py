import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

class OtimizacaoGrafoRoteamento(ElementwiseProblem):

    def __init__(self):
        # n_var=4 (Métricas dos Links de Acesso e Atributo de Borda)
        # n_obj=4 (Latência, Congestionamento, Estabilidade, Resiliência)
        # n_constr=4 (Restrições topológicas de capacidade e hierarquia)
        # Limites abstratos de configuração (ex: métricas de custo de 1.0 a 100.0)
        super().__init__(n_var=4, 
                         n_obj=4, 
                         n_constr=4, 
                         xl=np.array([1.0, 1.0, 1.0, 1.0]), 
                         xu=np.array([100.0, 100.0, 100.0, 100.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        # x[0] -> Métrica do Link_A (Acesso Inferior Esquerdo)
        # x[1] -> Métrica do Link_B (Acesso Superior Esquerdo)
        # x[2] -> Métrica do Link_C (Acesso de Agregação Intermediário)
        # x[3] -> Atributo de Preferência do Nó Central de Borda (Core)
        
        # --- FUNÇÕES OBJETIVO (Padronizadas para Minimização) ---
        
        # f1: Latência Média do Sistema (Ponderada pelo comprimento analítico dos caminhos)
        f1 = 0.15 * x[0] + 0.12 * x[1] + 0.18 * x[2] + 0.05 * x[3]
        
        # f2: Índice de Congestionamento (Função convexa baseada em Teoria de Filas)
        # Penaliza severamente custos excessivamente baixos que causam saturação nos links de menor capacidade
        f2 = (500 / (x[0] + 0.5)) + (400 / (x[1] + 0.5)) + (600 / (x[2] + 0.5)) + (x[3] * 0.1)
        
        # f3: Entropia de Estabilidade do Plano de Controle (Sobrecarga de CPU por convergência)
        # Evita variâncias extremas ou valores idênticos que geram micro-loops/oscilações de Dijkstra
        f3 = np.sum(x**2) * 0.005 + 10 / (np.std(x) + 0.1)
        
        # f4: Resiliência Topológica Ponderada (Invertida para Minimização)
        # Busca maximizar a dispersão e viabilidade de caminhos de backup
        f4 = -( (x[0] * 0.4) + (x[1] * 0.4) + (x[2] * 0.5) + (x[3] * 0.8) )

        # --- RESTRIÇÕES DE VIABILIDADE DO GRAFO (g <= 0) ---
        
        # g1: Restrição de Consistência Hierárquica (Borda deve manter precedência sobre acessos)
        g1 = x[0] + x[1] - x[3] 
        
        # g2: Limite de Capacidade Agregada dos Enlaces de Acesso (Evitar saturação física de uplink)
        g2 = (x[0] * 1.5) + (x[1] * 1.2) - 120
        
        # g3: Restrição de Tráfego de Trânsito Indevido no Nó Intermediário C
        g3 = 15.0 - x[2]
        
        # g4: Limite Mínimo de Custo de Caminho para Evitar Micro-loops Genéricos
        g4 = 20.0 - np.sum(x)

        out["F"] = [f1, f2, f3, f4]
        out["G"] = [g1, g2, g3, g4]

# 1. Configuração de Direções de Referência usando "das-dennis" para evitar estouro de memória
# Com 4 objetivos e 5 partições, geramos 56 direções bem distribuídas espacialmente
ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=5)

# 2. Inicialização da Meta-heurística NSGA-III
algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)

# 3. Execução do Algoritmo de Otimização
problem = OtimizacaoGrafoRoteamento()
res = minimize(problem, algorithm, termination=('n_gen', 250), seed=42, verbose=False)

# 4. Extração e Reversão dos Objetivos do Fronte de Pareto
F = res.F
latencia = F[:, 0]
congestionamento = F[:, 1]
estabilidade_cpu = F[:, 2]
resiliencia = -F[:, 3] # Reversão para escala real de maximização

# 5. Visualização Gráfica Abstrata em 4D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Mapeamento: X=Latência, Y=Congestionamento, Z=Instabilidade de CPU, Cor=Resiliência
sc = ax.scatter(latencia, congestionamento, estabilidade_cpu, c=resiliencia, cmap='plasma', s=60, edgecolors='black', alpha=0.8)

ax.set_xlabel('Métrica Espacial de Latência', fontsize=11)
ax.set_ylabel('Fator de Congestionamento dos Enlaces', fontsize=11)
ax.set_zlabel('Sobrecarga de Processamento do Plano de Controle', fontsize=11)

cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Índice Global de Resiliência da Topologia (Maior é melhor)', fontsize=11)

plt.title("Otimização Multiobjetivo 4D: Engenharia de Tráfego em Grafos de Redes", fontsize=13, fontweight='bold')
plt.show()

print(f"Configurações ótimas do Fronte de Pareto calculadas com sucesso: {len(res.X)}")