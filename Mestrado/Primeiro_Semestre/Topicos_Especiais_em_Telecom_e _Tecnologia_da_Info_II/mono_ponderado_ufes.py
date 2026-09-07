import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
import matplotlib.pyplot as plt

print("=== Inicializando Otimização Monoobjetivo Ponderada ===")

# ==============================================================================
# 1. BASES DE DADOS EXPERIMENTAIS (LOOKUP TABLES)
# ==============================================================================
pontos_dados_f2 = np.array([
    [0.5, 1.0], [2.0, 1.0], [4.0, 1.0], 
    [0.5, 3.0], [2.0, 3.0], [4.0, 3.0],
    [0.5, 5.0], [2.0, 5.0], [4.0, 5.0]
])
valores_dados_f2 = np.array([12.4, 25.1, 48.0, 18.2, 33.7, 58.1, 24.5, 42.9, 72.3])
interpolador_f2 = LinearNDInterpolator(pontos_dados_f2, valores_dados_f2)

dados_x2_restricao = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
limiares_interferencia = np.array([1.5, 2.8, 4.9, 7.2, 9.8])
interpolador_g2 = interp1d(dados_x2_restricao, limiares_interferencia, kind='linear', fill_value="extrapolate")

# Configurações do AG Interno
POP_SIZE = 40
GERACOES = 50
MUT_RATE = 0.1
LB = np.array([0.5, 1.0, 1.0])
UB = np.array([4.0, 5.0, 3.0])

# ==============================================================================
# 2. AVALIAÇÃO DO SISTEMA (CÁLCULO INDIVIDUAL DE F1 E F2)
# ==============================================================================
def calcular_objetivos_e_restricoes(x):
    x1, x2, x3 = x[0], x[1], x[2]
    
    # f1 (Equação) e f2 (Dados)
    f1 = (5.0 / (x1 * x2)) + (x3 - 2.2)**2
    f2 = float(interpolador_f2(x1, x2)) + 0.6 * x3
    
    # g1 (Equação) e g2 (Dados)
    g1 = 1.2 * x1 + 1.8 * x2 + x3 - 9.5
    g2 = (x1 * x3) - float(interpolador_g2(x2))
    
    # Penalização severa se violar restrições
    penalidade = 0.0
    if g1 > 0: penalidade += g1 * 1000
    if g2 > 0: penalidade += g2 * 1000
        
    return f1, f2, penalidade

# ==============================================================================
# 3. ALGORITMO GENÉTICO MONOOBJETIVO (OTIMIZA POR PESO ALPHA)
# ==============================================================================
def otimizar_monoobjetivo(alpha):
    # População inicial aleatória
    X = np.random.uniform(LB, UB, (POP_SIZE, 3))
    
    for gen in range(GERACOES):
        # Avaliar a função ponderada para a população atual
        fitness = []
        for ind in X:
            f1, f2, pen = calcular_objetivos_e_restricoes(ind)
            # Função Objetivo Ponderada + Penalidade de Restrição
            f_mono = alpha * f1 + (1 - alpha) * f2 + pen
            fitness.append(f_mono)
        fitness = np.array(fitness)
        
        # Seleção dos melhores (Elitismo)
        idx_ordenado = np.argsort(fitness)
        X = X[idx_ordenado] # Ordena a população do melhor pro pior
        
        # Gerar Filhos para substituir a metade pior
        X_filhos = []
        while len(X_filhos) < POP_SIZE // 2:
            # Seleção dos pais entre a metade superior
            idx_p1, idx_p2 = np.random.choice(POP_SIZE // 2, 2, replace=False)
            p1, p2 = X[idx_p1], X[idx_p2]
            
            # Cruzamento ponderado aleatório
            w = np.random.rand(3)
            filho = w * p1 + (1 - w) * p2
            
            # Mutação
            if np.random.rand() < MUT_RATE:
                filho += np.random.normal(0, 0.05, 3)
                
            filho = np.clip(filho, LB, UB)
            X_filhos.append(filho)
            
        X[POP_SIZE // 2:] = np.array(X_filhos)
        
    # Retorna o melhor indivíduo da última geração e seus valores reais de f1 e f2
    melhor_ind = X[0]
    f1_otimo, f2_otimo, _ = calcular_objetivos_e_restricoes(melhor_ind)
    return f1_otimo, f2_otimo

# ==============================================================================
# 4. VARREDURA DOS PESOS ALFA PARA GERAR A FRONTEIRA
# ==============================================================================
# Vamos testar 50 valores de alpha de 0.001 a 0.999
valores_alpha = np.linspace(0.001, 0.999, 50)
fronteira_f1 = []
fronteira_f2 = []

print("-> Varrendo os pesos normativos de Alpha...")
for alpha in valores_alpha:
    f1, f2 = otimizar_monoobjetivo(alpha)
    fronteira_f1.append(f1)
    fronteira_f2.append(f2)

# ==============================================================================
# 5. PLOTAGEM DA CURVA DE PARETO PONDERADA
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.plot(fronteira_f1, fronteira_f2, color='#6366f1', linestyle='-', alpha=0.3)
plt.scatter(fronteira_f1, fronteira_f2, color='#4f46e5', edgecolors='k', s=45, label='Pontos Ótimos Ponderados')
plt.title('Fronteira de Pareto via Soma Ponderada (Monoobjetivo)', fontsize=11, fontweight='bold')
plt.xlabel('f1: Métrica via Equação (Mínimo)')
plt.ylabel('f2: Métrica via Base de Dados (Mínimo)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.savefig('pareto_mono_ponderado.png', dpi=300, bbox_inches='tight')
plt.close()
print("[SUCESSO] Varredura concluída. Gráfico salvo como 'pareto_mono_ponderado.png'")