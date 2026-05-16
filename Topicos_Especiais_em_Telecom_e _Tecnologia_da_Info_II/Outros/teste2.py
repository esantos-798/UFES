import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

class BGPProblemFinal(Problem):
    def __init__(self):
        super().__init__(n_var=8, n_obj=4, n_ieq_constr=10, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # Objetivos
        f1 = x[:, 0]*0.5 + x[:, 5]*0.2  # Latência
        f2 = x[:, 1]*0.8 + x[:, 2]*0.3  # Custo
        f3 = -(x[:, 3]*0.6 + x[:, 7]*0.4) # Resiliência
        f4 = x[:, 4]*0.7 + x[:, 6]*0.3  # Congestão
        out["F"] = np.column_stack([f1, f2, f3, f4])

        # Restrições (Relaxadas para evitar conjunto vazio)
        g = np.zeros((x.shape[0], 10))
        g[:, 0] = 0.01 - x[:, 0]
        g[:, 1] = x[:, 1] - 0.99
        out["G"] = g

# Instanciar o problema
problem = BGPProblemFinal()
algorithm = NSGA2(pop_size=50)

try:
    # Execução (copy_algorithm=False é vital no Python 3.13)
    res = minimize(problem, algorithm, termination=('n_gen', 50), copy_algorithm=False)

    if res.F is not None and len(res.F) > 0:
        data = res.F.copy()
        data[:, 2] = -data[:, 2] # Restaurar valor positivo
        
        df = pd.DataFrame(data, columns=['Latencia', 'Custo', 'Resiliencia', 'Congestao'])
        
        # MUDANÇA AQUI: Nomes de legenda sem '$' puro para evitar erro de LaTeX no Matplotlib
        df['Categoria_Custo'] = pd.qcut(df['Custo'], q=3, labels=['Economico', 'Standard', 'Premium'])

        plt.figure(figsize=(10, 6))
        # Plotando sem as legendas automáticas que causam conflito
        parallel_coordinates(df, 'Categoria_Custo', colormap='viridis', alpha=0.6)
        
        plt.title('Otimização Multiobjetivo BGP')
        plt.ylabel('Score (0-1)')
        plt.show()
    else:
        print("Nenhuma solução encontrada.")

except Exception as e:
    print(f"Erro inesperado: {e}")