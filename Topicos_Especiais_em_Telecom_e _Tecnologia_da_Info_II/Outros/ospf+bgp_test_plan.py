import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

class OtimizacaoTesteSistemico(ElementwiseProblem):

    def __init__(self):
        # n_var=4 (Tipos de Teste), n_obj=4, n_constr=3
        # Limites: Cada tipo de teste pode rodar de 0 a 5 horas na janela de laboratório
        super().__init__(n_var=4, 
                         n_obj=4, 
                         n_constr=3, 
                         xl=np.zeros(4), 
                         xu=np.array([5.0, 5.0, 5.0, 5.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Coeficientes fictícios de impacto (Pesos baseados na criticidade real do ambiente)
        # x[0]=Plano de Controle, x[1]=Plano de Dados, x[2]=Resiliência, x[3]=L2VPN/VPLS
        
        # --- FUNÇÕES OBJETIVO ---
        
        # f1: Maximizar Cobertura de Riscos (Invertido para minimizar)
        # Testar L2VPN/VPLS (x[3]) e Resiliência (x[2]) mitiga a maior parte dos riscos da rede de Cariacica
        f1 = -(0.3 * x[0] + 0.2 * x[1] + 0.5 * x[2] + 0.6 * x[3])
        
        # f2: Minimizar Tempo Total de Execução do Teste (Soma simples do tempo alocado)
        f2 = np.sum(x)
        
        # f3: Minimizar Risco de Travamento do Equipamento de Teste/Ambiente (Stress acumulado não-linear)
        f3 = 0.1 * (x[0]**2) + 0.05 * (x[1]**2) + 0.2 * (x[2] * x[3])
        
        # f4: Maximizar Diversidade de Cenários (Invertido para minimizar)
        # Penaliza planos de teste que zeram ou ignoram completamente uma das frentes
        f4 = -float(np.min(x) + 0.1 * np.std(x))

        # --- RESTRIÇÕES (g <= 0) ---
        # g1: Janela máxima de tempo disponível no laboratório/Spirent não pode passar de 12 horas
        g1 = np.sum(x) - 12.0
        
        # g2: O teste de plano de dados (throughput) precisa rodar por pelo menos 45 minutos (0.75h) para coletar métricas de RFC2544 estáveis
        g2 = 0.75 - x[1]
        
        # g3: O tempo de teste de L2VPN/VPLS não pode ser inferior ao de resiliência, pois dependem um do outro na topologia
        g3 = x[2] - x[3]

        out["F"] = [f1, f2, f3, f4]
        out["G"] = [g1, g2, g3]

# 1. Configurando o Algoritmo Genético
ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=5)
algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)

# 2. Resolvendo o problema
problem = OtimizacaoTesteSistemico()
res = minimize(problem, algorithm, termination=('n_gen', 200), seed=1, verbose=False)

# 3. Extraindo as frentes de Pareto
F = res.F
cobertura_risco = -F[:, 0]  # Reverte para valor positivo (Maximizar)
tempo_total = F[:, 1]
risco_infra = F[:, 2]
diversidade = -F[:, 3]      # Reverte para valor positivo (Maximizar)

# 4. Plotando o Gráfico 4D do Planejamento de Testes
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

# X: Tempo de Janela, Y: Cobertura de Riscos, Z: Stress da Infra, Cor: Diversidade dos Testes
sc = ax.scatter(tempo_total, cobertura_risco, risco_infra, c=diversidade, cmap='coolwarm', s=60, edgecolors='k')

ax.set_xlabel('Tempo Total de Janela (Horas)')
ax.set_ylabel('Cobertura de Riscos de Rede')
ax.set_zlabel('Índice de Stress/Risco da Infra de Teste')

cbar = plt.colorbar(sc)
cbar.set_label('Índice de Diversidade do Plano de Teste (Maior é melhor)')

plt.title("Otimização Multiobjetivo 4D: Planejamento de Teste Sistêmico", fontsize=12, fontweight='bold')
plt.show()

print(f"Planos de teste ótimos gerados pelo NSGA-III: {len(res.X)}")