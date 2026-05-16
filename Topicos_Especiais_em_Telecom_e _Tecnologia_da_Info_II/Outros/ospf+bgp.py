import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

class OtimizacaoRedeCariacica(ElementwiseProblem):

    def __init__(self):
        # n_var=4 (DM2500x2, DM2514, DM4270), n_obj=4, n_constr=4
        # Limites das métricas: OSPF varia de 1 a 100 nesta modelagem simplificada
        super().__init__(n_var=4, 
                         n_obj=4, 
                         n_constr=4, 
                         xl=np.array([1.0, 1.0, 1.0, 1.0]), 
                         xu=np.array([100.0, 100.0, 100.0, 100.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Parâmetros simulados baseados na capacidade física dos equipamentos da imagem
        # DM2500 (1G/GE), DM2514 (1G/GE), DM4270 (10G/100G)
        
        # --- FUNÇÕES OBJETIVO (pymoo busca sempre a MINIMIZAÇÃO) ---
        
        # f1: Latência Média - proporcional ao valor das métricas escolhidas (caminhos mais longos = mais saltos)
        f1 = 0.15 * x[0] + 0.12 * x[1] + 0.18 * x[2] + 0.05 * x[3]
        
        # f2: Utilização Máxima (Congestionamento) - se a métrica for muito baixa, satura o link de acesso (DM2500)
        # É uma função não-linear (penaliza severamente métricas excessivamente baixas que causam gargalos)
        f2 = (500 / (x[0] + 0.5)) + (400 / (x[1] + 0.5)) + (600 / (x[2] + 0.5)) + (x[3] * 0.1)
        
        # f3: Sobrecarga de CPU devido ao processamento OSPF/BGP
        # Relação matemática instável: métricas muito altas ou muito próximas geram instabilidade de convergência
        f3 = np.sum(x**2) * 0.005 + 10 / (np.std(x) + 0.1)
        
        # f4: Maximizar Resiliência (Invertido para Minimizar)
        # Queremos maximizar a dispersão e existência de caminhos secundários estáveis
        f4 = -( (x[0] * 0.4) + (x[1] * 0.4) + (x[2] * 0.5) + (x[3] * 0.8) )

        # --- RESTRIÇÕES (g <= 0) ---
        # g1: O DM4270 (Core) deve ter uma preferência/métrica hierarquicamente estruturada em relação aos acessos
        g1 = x[0] + x[1] - x[3] 
        
        # g2: Limite de tráfego agregado projetado no mesh DM2500 não pode estourar as interfaces de UpLink (Ex: g-1/1/2)
        g2 = (x[0] * 1.5) + (x[1] * 1.2) - 120
        
        # g3: Garantia que o DM2514 possua métrica suficiente para não virar tráfego de trânsito indevido
        g3 = 15.0 - x[2]
        
        # g4: Estabilidade de convergência (Métrica acumulada mínima para evitar micro-loops)
        g4 = 20.0 - np.sum(x)

        out["F"] = [f1, f2, f3, f4]
        out["G"] = [g1, g2, g3, g4]

# 1. Configuração do NSGA-III (Apropriado para 4 objetivos)
ref_dirs = get_reference_directions("energy", 4, n_points=91)
algorithm = NSGA3(pop_size=120, ref_dirs=ref_dirs)

# 2. Execução
problem = OtimizacaoRedeCariacica()
res = minimize(problem, algorithm, termination=('n_gen', 250), seed=42, verbose=False)

# 3. Extração dos Resultados para o Fronte de Pareto
F = res.F
latencia = F[:, 0]
congestionamento = F[:, 1]
cpu_overhead = F[:, 2]
resiliencia = -F[:, 3] # Inverte para exibir o valor real da maximização

# 4. Construção do Gráfico 4D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Mapeando os 4 objetivos: X=Latência, Y=Congestionamento, Z=Sobrecarga de CPU, Cor=Resiliência
sc = ax.scatter(latencia, congestionamento, cpu_overhead, c=resiliencia, cmap='plasma', s=60, edgecolors='black', alpha=0.8)

ax.set_xlabel('Latência da Rede (ms)', fontsize=11)
ax.set_ylabel('Índice de Congestionamento (Links)', fontsize=11)
ax.set_zlabel('Sobrecarga de CPU nos Roteadores', fontsize=11)

cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Nível de Resiliência da Topologia (Maior é melhor)', fontsize=11)

plt.title("Otimização 4D de Engenharia de Tráfego (OSPF+BGP) - Mesh Cariacica", fontsize=14, fontweight='bold')
plt.show()

print(f"Número de configurações ótimas (Fronte de Pareto) encontradas: {len(res.X)}")