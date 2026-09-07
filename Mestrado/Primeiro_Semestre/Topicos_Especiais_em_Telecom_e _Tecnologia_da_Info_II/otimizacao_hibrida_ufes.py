import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
import matplotlib.pyplot as plt

print("=== Inicializando Problema Multiobjetivo (Equações + Dados) ===")

# ==========================================
# 1. SIMULAÇÃO DOS DADOS DE LABORATÓRIO (LOOKUP TABLES)
# ==========================================
# Dados para o Objetivo 2 (f2): Custo baseado em x1 (potência) e x2 (banda)
pontos_x1_x2 = np.array([[0.5, 1.0], [2.0, 1.0], [4.0, 1.0], 
                         [0.5, 5.0], [2.0, 5.0], [4.0, 5.0]])
valores_custo = np.array([10.5, 22.1, 45.0, 15.3, 30.2, 60.8])
# Interpolação 2D para ler os dados do Objetivo 2
lookup_f2_dados = LinearNDInterpolator(pontos_x1_x2, valores_custo)

# Dados para a Restrição 2 (g2): Limiar de interferência baseado em x2 (banda)
dados_banda_x2 = np.array([1.0, 2.5, 4.0, 5.0])
dados_interferencia = np.array([1.2, 3.5, 6.8, 9.5])
# Interpolação 1D para ler os dados da Restrição 2
lookup_g2_dados = interp1d(dados_banda_x2, dados_interferencia, kind='linear', fill_value="extrapolate")

# ==========================================
# 2. DEFINIÇÃO DO PROBLEMA NO PYMOO
# ==========================================
class ProblemaHibridoUFES(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=3,   # 3 variáveis de decisão (x1, x2, x3)
            n_obj=2,   # 2 funções objetivo (f1, f2)
            n_ieq_constr=2, # 2 restrições (g1, g2)
            xl=np.array([0.5, 1.0, 1.0]), # Limites inferiores
            xu=np.array([4.0, 5.0, 3.0])  # Limites superiores
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x1, x2, x3 = x[0], x[1], x[2]

        # --- FUNÇÕES OBJETIVO ---
        # f1: Totalmente baseada em Equação
        f1 = (4.0 / (x1 * x2)) + (x3 - 2.5)**2
        
        # f2: Baseada em Dados + componente linear
        custo_base = float(lookup_f2_dados(x1, x2))
        f2 = custo_base + 0.5 * x3

        # --- RESTRIÇÕES (Padrão pymoo: devem ser <= 0 se válidas) ---
        # g1: Totalmente baseada em Equação
        g1 = 1.5*x1 + 2.0*x2 + x3 - 10.0
        
        # g2: Baseada em Dados
        limiar_interf = float(lookup_g2_dados(x2))
        g2 = (x1 * x3) - limiar_interf

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

# ==========================================
# 3. EXECUÇÃO DA OTIMIZAÇÃO (NSGA-II)
# ==========================================
problem = ProblemaHibridoUFES()
algorithm = NSGA2(pop_size=50)

print("-> Executando o algoritmo evolucionário NSGA-II...")
res = minimize(problem, algorithm, ('n_gen', 100), seed=42, verbose=False)

# ==========================================
# 4. PLOTAGEM DA FRONTEIRA DE PARETO
# ==========================================
print(f"-> Otimização concluída! Soluções ótimas encontradas: {len(res.F)}")

plt.figure(figsize=(8, 5))
plt.scatter(res.F[:, 0], res.F[:, 1], color='#38bdf8', edgecolors='k', s=50, label='Fronteira de Pareto')
plt.title('Fronteira de Pareto: Equações vs. Dados (UFES 2026)', fontsize=12, fontweight='bold')
plt.xlabel('Objetivo 1: Erro de Sinal (f1 - Mínimo)', fontsize=10)
plt.ylabel('Objetivo 2: Custo por Dados (f2 - Mínimo)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

grafico_nome = 'fronteira_pareto_hibrida.png'
plt.savefig(grafico_nome, dpi=300, bbox_inches='tight')
plt.close()
print(f"[SUCESSO] Gráfico salvo como '{grafico_nome}'")