import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination import get_termination

class ProducaoSustentavel(ElementwiseProblem):

    def __init__(self):
        # n_var=8, n_obj=4, n_ieq_constr=9 (R1 a R9, R10 é tratada nos limites xl/xu)
        super().__init__(n_var=8, 
                         n_obj=4, 
                         n_constr=9, 
                         xl=np.zeros(8), 
                         xu=np.array([10.0]*8)) # Limite superior de 10k unidades por tipo

    def _evaluate(self, x, out, *args, **kwargs):
        # Parâmetros fictícios para o exemplo
        p = np.array([50, 60, 45, 70, 55, 65, 80, 85])  # Lucro
        e = np.array([2.1, 3.5, 1.8, 4.2, 2.5, 3.0, 5.0, 4.8]) # Emissões
        s = np.array([0.5, 0.4, 0.6, 0.3, 0.7, 0.5, 0.4, 0.6]) # Setup cost coef
        q = np.array([0.95, 0.90, 0.98, 0.88, 0.92, 0.94, 0.97, 0.96]) # Qualidade
        
        # Coeficientes das restrições (ex: mão de obra, matéria-prima, etc)
        a = np.array([10, 12, 8, 15, 10, 11, 14, 16]) # R1
        b = np.array([1.2, 1.5, 0.8, 2.0, 1.1, 1.3, 1.7, 1.9]) # R2

        # --- FUNÇÕES OBJETIVO ---
        f1 = -(np.sum(p * x) - 500) # Negativo para Maximizar Lucro
        f2 = np.sum(e * x)          # Minimizar Emissões
        f3 = np.sum(s * (x**2))     # Minimizar Tempo de Setup
        f4 = -(np.sum(q * x) / (np.sum(x) + 1e-6)) # Maximizar Qualidade Média

        # --- RESTRIÇÕES (g <= 0) ---
        g1 = np.sum(a * x) - 40000             # R1: Mão de obra
        g2 = np.sum(b * x) - 5000              # R2: Matéria-prima
        g3 = 1.5 - (x[0] + x[1] + x[2])        # R3: Demanda Mínima (inverte sinal)
        g4 = np.sum(1.2 * x) - 10000           # R4: Armazenagem (v_i simplificado)
        g5 = np.sum(25 * x) - 200000           # R5: Orçamento (c_i simplificado)
        g6 = x[4] - 2*x[3]                     # R6: Equilíbrio de linha
        g7 = 0.8 - (x[6] + x[7])               # R7: Exportação
        g8 = np.sum(0.1 * x) - 300             # R8: Resíduos
        g9 = np.sum(5 * x) - 15000             # R9: Energia

        out["F"] = [f1, f2, f3, f4]
        out["G"] = [g1, g2, g3, g4, g5, g6, g7, g8, g9]

# 1. Definir direções de referência (Crucial para NSGA-III)
ref_dirs = get_reference_directions("energy", 4, n_points=91)

# 2. Inicializar o algoritmo
algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)

# 3. Executar a otimização
problem = ProducaoSustentavel()
res = minimize(problem,
               algorithm,
               termination=('n_gen', 200),
               seed=1,
               verbose=True)

print(f"Soluções encontradas no Fronte de Pareto: {len(res.X)}")

import matplotlib.pyplot as plt

# Extrair os objetivos das soluções encontradas
F = res.F

# f1: Lucro (invertemos para ficar positivo)
# f2: Emissões
# f3: Setup
# f4: Qualidade (invertemos para ficar positivo)
lucro = -F[:, 0]
emissoes = F[:, 1]
setup = F[:, 2]
qualidade = -F[:, 3]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# O parâmetro 'c' define a 4ª dimensão através da cor
sc = ax.scatter(lucro, emissoes, setup, c=qualidade, cmap='viridis', s=50)

ax.set_xlabel('Lucro')
ax.set_ylabel('Emissões')
ax.set_zlabel('Tempo de Setup')
plt.colorbar(sc, label='Qualidade Média')
plt.title("Visualização 4D: Posição (3D) + Cor (Qualidade)")
plt.show()