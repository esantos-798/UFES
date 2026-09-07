import numpy as np
import matplotlib.pyplot as plt

def run_haf_tuned(obj_func):
    # -------------------------------------------------------------------------
    # 1. INICIALIZAÇÃO DO ECOSSISTEMA HIDRÁULICO
    # -------------------------------------------------------------------------
    # Criamos a matriz de posições. Cada linha é uma partícula fluida (solução).
    # Cada coluna representa uma dimensão (variável do problema).
    X = np.random.uniform(bounds[0], bounds[1], size=(num_pop, D))
    
    # Matriz de Velocidades: Guarda a inércia (direção e módulo) do movimento anterior.
    # Inicialmente está zerada porque o fluido está estático no tempo t=0.
    V = np.zeros((num_pop, D))
    
    # Vetor que armazena a coordenada do "Ralo" (o ponto de menor pressão/melhor custo).
    best_pos = np.zeros(D)
    best_score = float('inf') # Começa no infinito para que qualquer solução seja melhor
    
    # Histórico para plotar o gráfico de convergência no final
    history = []
    
    # Contadores de estagnação para medir se o fluido ficou preso em um redemoinho
    cont_estagnacao = 0
    last_best = float('inf')

    # -------------------------------------------------------------------------
    # 2. LOOP TEMPORAL (ITERAÇÕES DO FLUXO)
    # -------------------------------------------------------------------------
    for t in range(max_iter):
        
        # --- ETAPA DE ELITISMO (Mapeamento do Ponto de Pressão Mínima) ---
        # Varremos toda a população para encontrar quem achou o vale mais profundo.
        for i in range(num_pop):
            score = obj_func(X[i]) # Calcula a altitude/custo da partícula
            
            if score < best_score:
                best_score = score
                best_pos = np.copy(X[i]) # O Elitismo grava a coordenada de ouro
                
        history.append(best_score) # Registra para o gráfico
        
        # --- MONITOR DE PRESSÃO (Detecção de Mínimos Locais) ---
        # Se o melhor score não mudou em relação à iteração anterior, a pressão acumula.
        if best_score == last_best:
            cont_estagnacao += 1
        else:
            cont_estagnacao = 0 # Se melhorou, o fluido está escorrendo bem, reseta o alarme
            last_best = best_score
        
        # --- AJUSTE DE PROPRIEDADES FÍSICAS DO FLUIDO (Parâmetros Adaptativos) ---
        # Viscosidade decai linearmente: o fluido começa como gás (muito volátil, varre o mapa)
        # e termina espesso como mel (move-se pouco, focado em refinar onde está).
        viscosidade = 0.7 * (1.0 - (t / max_iter)) 
        
        # Sucção magnética aumenta: a força de atração do líder fica mais violenta no final.
        succao = 0.2 + 0.6 * (t / max_iter)        
        
        # --- OPERADOR DE TURBULÊNCIAS CONTRA REDEMOINHOS DE GRADIENTE ---
        # IMPORTANTE: Em 10 dimensões, as partículas acumulam velocidades conflitantes 
        # nas colunas, criando um "redemoinho". Elas orbitam o erro e não conseguem sair.
        if cont_estagnacao > 3:
            # O alarme disparou! Forçamos uma quebra de simetria hidráulica.
            for i in range(num_pop):
                if np.random.rand() < 0.5:
                    # PASSO CRÍTICO: Zeramos a velocidade acumulada (V[i]=0).
                    # Isso "esvazia" a memória do erro e para o redemoinho instantaneamente.
                    V[i] = np.zeros(D) 
                    
                    # Teletransportamos a partícula para a vizinhança do líder, mas com
                    # um ruído muito curto (entre -0.1 e 0.1). Isso força o fluido a
                    # procurar frestas e canais capilares colados ao melhor ponto atual.
                    X[i] = best_pos + np.random.uniform(-0.1, 0.1, size=D)
            
            cont_estagnacao = 0 # Reseta o medidor após a purga de pressão
            
        # --- ATUALIZAÇÃO DA DINÂMICA DE MOVIMENTO (Equação de Navier-Stokes) ---
        for i in range(num_pop):
            # r1 é um escalar aleatório. Ele dita o "ritmo" da puxada, mas preserva 
            # a direção geométrica exata do vetor de gradiente (linha reta até o líder).
            r1 = np.random.rand()
            
            # O vetor gradiente mede a distância e a direção da partícula até o ralo.
            grad_pressao = best_pos - X[i]
            
            # Nova Velocidade = (Inércia do movimento passado) + (Força de sucção ao líder)
            # Se a viscosidade está alta, a partícula ignora um pouco o líder e passa direto.
            # Se a sucção está alta, ela faz uma curva fechada em direção ao best_pos.
            V[i] = viscosidade * V[i] + succao * r1 * grad_pressao
            
            # Nova Posição = Posição Atual + Velocidade Calculada
            X[i] = X[i] + V[i]
            
            # --- PROTEÇÃO DE FRONTEIRA GEOGRÁFICA ---
            # Se o fluido "vazar" para fora das paredes do benchmark (ex: passar de 5.12),
            # o clip rebate a partícula, travando-a na borda do mapa.
            X[i] = np.clip(X[i], bounds[0], bounds[1])
            
    return best_score, history