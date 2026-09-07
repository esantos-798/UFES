import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

class OSPFWeightOptimization(Problem):
    def __init__(self):
        # n_var = 8 (pesos de 8 links principais da rede)
        # n_obj = 4 (Métricas de performance da rede)
        # n_ieq_constr = 2 (Restrições operacionais)
        # No OSPF, os pesos variam de 1 a 65535, mas vamos normalizar de 1.0 a 100.0 para a metaeurística
        super().__init__(n_var=8, n_obj=4, n_ieq_constr=2, xl=1.0, xu=100.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # x representa a matriz de populações pelas variáveis (pesos dos links)
        
        # 1. Congestão (MLU) - Simulado: Pesos muito baixos em certos links geram gargalos
        f1 = (100.0 / (x[:, 0] + x[:, 1] + 1e-5)) + x[:, 4] * 0.4
        
        # 2. Atraso Total (Delay) - Simulado: Pesos mais altos fazem o OSPF escolher caminhos mais longos
        f2 = x[:, 2] * 0.5 + x[:, 3] * 0.6 + x[:, 5] * 0.3
        
        # 3. Custo de Operação Financeira dos Links
        f3 = x[:, 0] * 0.2 + x[:, 6] * 0.7
        
        # 4. Resiliência da Rede (Queremos MAXIMIZAR, então multiplicamos por -1 para o pymoo minimizar)
        # Aqui simulamos que links 1, 5 e 7 são os mais estáveis.
        f4 = -(x[:, 1] * 0.4 + x[:, 5] * 0.3 + x[:, 7] * 0.3)
        
        # Agrupando os objetivos
        out["F"] = np.column_stack([f1, f2, f3, f4])

        # Restrições (g <= 0)
        g = np.zeros((x.shape[0], 2))
        # Exemplo: O peso do Link 0 não pode ser drasticamente menor que o do Link 2 (para evitar oscilação de rotas)
        g[:, 0] = (x[:, 2] - x[:, 0]) - 80.0 
        # Exemplo: Atraso simulado (f2) não pode estourar um limite crítico de 120
        g[:, 1] = f2 - 120.0
        
        out["G"] = g

# Instanciar o problema do OSPF
problem = OSPFWeightOptimization()
algorithm = NSGA2(pop_size=50)

try:
    # Execução segura para Python 3.13+
    res = minimize(problem, algorithm, termination=('n_gen', 50), copy_algorithm=False)

    if res.F is not None and len(res.F) > 0:
        data = res.F.copy()
        data[:, 3] = -data[:, 3] # Restaurar o valor positivo da Resiliência
        
        # DataFrame para visualização
        df = pd.DataFrame(data, columns=['Congestao_MLU', 'Delay', 'Custo_Op', 'Resiliencia'])
        
        # Criando categorias baseadas no Delay para a legenda do gráfico
        df['Categoria_Delay'] = pd.qcut(df['Delay'], q=3, labels=['Baixo_Atraso', 'Moderado', 'Alto_Atraso'])

        # Plot das Coordenadas Paralelas
        plt.figure(figsize=(11, 6))
        parallel_coordinates(df, 'Categoria_Delay', colormap='plasma', alpha=0.7)
        
        plt.title('Otimização Multiobjetivo de Pesos OSPF (Engenharia de Tráfego)')
        plt.ylabel('Score / Métrica')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
        
        # Exibir uma amostra das soluções da Fronteira de Pareto
        print("Amostra das melhores configurações encontradas (Fronteira de Pareto):")
        print(df.head())
    else:
        print("Nenhuma solução válida foi encontrada dentro das restrições.")

except Exception as e:
    print(f"Erro inesperado na execução: {e}")