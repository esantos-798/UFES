"""
==============================================================================
PROJETO - Identificação de Sistemas | UFES 2026
Tema: Predição de Atraso e Detecção de Falhas em Redes 5G/OTN

SCRIPT 2 de 3: Identificação Recursiva (RLS) + Modelo ARX
==============================================================================

CONTEXTO:
    Neste script fingimos que NÃO conhecemos os parâmetros internos da fila
    (a1, b1). Temos apenas os dados medidos: u[k] e y[k] do CSV gerado pelo
    Script 1. O algoritmo RLS vai "aprender" esses parâmetros observando
    apenas os dados, instante a instante.

MODELO ARX (1ª ordem):
    y[k] = a1 * y[k-1] + b1 * u[k-1] + e[k]

    Em forma vetorial (como o RLS enxerga):
    y[k] = phi[k]^T * theta + e[k]

    onde:
        phi[k]  = [y[k-1], u[k-1]]   <- Vetor de regressores (dados conhecidos)
        theta   = [a1, b1]            <- Vetor de parâmetros (o que queremos achar)
        e[k]    = erro/ruído

ALGORITMO RLS COM FATOR DE ESQUECIMENTO (lambda):
    A cada novo instante k, o RLS atualiza theta com 3 equações:

    1. Ganho de Kalman:
       K[k] = P[k-1] * phi[k] / (lambda + phi[k]^T * P[k-1] * phi[k])

    2. Atualização dos parâmetros:
       theta[k] = theta[k-1] + K[k] * (y[k] - phi[k]^T * theta[k-1])
                                        |__________________________|
                                               Inovação / Resíduo r[k]

    3. Atualização da covariância:
       P[k] = (1/lambda) * (I - K[k] * phi[k]^T) * P[k-1]

    O FATOR DE ESQUECIMENTO (lambda):
        - lambda = 1.0  -> Pesos iguais para todos os dados (MQ clássico)
        - lambda < 1.0  -> Dados recentes têm mais peso (acompanha variações)
        - Usaremos lambda = 0.98 para rastrear a mudança causada pela falha

==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Backend sem janela interativa (evita travamento)
import matplotlib.pyplot as plt

# =============================================================================
# 1. CARREGAR OS DADOS DO SCRIPT 1
# =============================================================================

df = pd.read_csv('dados_simulacao.csv')

t  = df['tempo_s'].values       # Vetor de tempo [s]
u  = df['trafego_u'].values     # Entrada: trafego [Gbps]
y  = df['atraso_y'].values      # Saida: atraso medido [ms]
N  = len(t)
TS = t[1] - t[0]                # Periodo de amostragem

INSTANTE_FALHA = 30.0           # Apenas para marcar nos graficos

print(f"Dados carregados: {N} amostras | Ts = {TS*1000:.0f} ms")

# =============================================================================
# 2. PARAMETROS DO ALGORITMO RLS
# =============================================================================

LAMBDA      = 0.98      # Fator de esquecimento
                        # 0.98 = janela de memoria ~50 amostras (0.5s)
                        # Diminuir -> rastreia mais rapido, mas mais ruidoso
                        # Aumentar -> mais suave, mas reage mais devagar a falhas

P0_DIAG     = 1e4       # Covariancia inicial (alta = muita incerteza inicial)
                        # O RLS vai reduzindo P[k] conforme aprende

# =============================================================================
# 3. INICIALIZACAO DO RLS
# =============================================================================

# Vetor de parametros inicial: chute inicial [a1=0, b1=0]
# O RLS vai convergir para os valores corretos nos primeiros instantes
theta = np.array([0.0, 0.0])

# Matriz de covariancia inicial (2x2, pois temos 2 parametros)
P = P0_DIAG * np.eye(2)

# Vetores para guardar o historico de cada variavel ao longo do tempo
historico_a1       = np.zeros(N)   # Parametro a1 estimado em cada instante
historico_b1       = np.zeros(N)   # Parametro b1 estimado em cada instante
historico_residuo  = np.zeros(N)   # Residuo r[k] = y[k] - y_predito[k]
historico_y_pred   = np.zeros(N)   # Predicao 1-passo a frente y_hat[k]

# =============================================================================
# 4. LOOP PRINCIPAL DO RLS (instante a instante)
# =============================================================================

print("Executando RLS...")

for k in range(1, N):

    # --- Monta o vetor de regressores com os dados do instante ANTERIOR ---
    phi = np.array([y[k-1],   # y[k-1]: atraso no instante anterior
                    u[k-1]])  # u[k-1]: trafego no instante anterior

    # --- Predicao: quanto o modelo ATUAL acha que sera o atraso agora ---
    y_pred = phi @ theta      # phi^T * theta  (produto escalar)
    historico_y_pred[k] = y_pred

    # --- Residuo (Inovacao): diferenca entre o real e o predito ---
    # Este e o numero mais importante do projeto:
    #   - Proximo de 0  -> modelo acertou a predicao
    #   - Longe de 0    -> houve uma surpresa (possivel falha!)
    r = y[k] - y_pred
    historico_residuo[k] = r

    # --- Equacao 1: Ganho de Kalman ---
    denominador = LAMBDA + phi @ P @ phi
    K = (P @ phi) / denominador

    # --- Equacao 2: Atualiza os parametros ---
    theta = theta + K * r

    # --- Equacao 3: Atualiza a matriz de covariancia ---
    P = (1 / LAMBDA) * (np.eye(2) - np.outer(K, phi)) @ P

    # --- Guarda historico ---
    historico_a1[k] = theta[0]
    historico_b1[k] = theta[1]

print(f"RLS concluido!")
print(f"  Parametros finais (apos convergencia):")
print(f"    a1 = {historico_a1[-1]:.4f}  (memoria da fila)")
print(f"    b1 = {historico_b1[-1]:.6f}  (impacto do trafego)")
print(f"  Residuo medio (ultimos 5s antes da falha):  "
      f"{np.mean(np.abs(historico_residuo[t < INSTANTE_FALHA][-500:])):.4f} ms")
print(f"  Residuo medio (primeiros 5s apos a falha):  "
      f"{np.mean(np.abs(historico_residuo[(t >= INSTANTE_FALHA) & (t < INSTANTE_FALHA+5)])):.4f} ms")

# =============================================================================
# 5. ANALISE DE RESIDUOS (Teste de Brancura - Valida se ARX e suficiente)
# =============================================================================
# Se o residuo for "branco" (descorrelacionado), o ARX capturou toda a
# dinamica do sistema. Se houver picos na autocorrelacao, precisamos do ARMAX.

residuo_estavel = historico_residuo[500:int(INSTANTE_FALHA/TS)]
autocorr = np.correlate(residuo_estavel - residuo_estavel.mean(),
                         residuo_estavel - residuo_estavel.mean(), mode='full')
autocorr = autocorr / autocorr.max()
lags = np.arange(-len(residuo_estavel)+1, len(residuo_estavel))

# Limite de significancia estatistica (95% de confianca)
limite_95 = 1.96 / np.sqrt(len(residuo_estavel))

# =============================================================================
# 6. SALVAR RESULTADOS EM CSV (para uso no Script 3)
# =============================================================================

df_resultado = df.copy()
df_resultado['a1_estimado']  = historico_a1
df_resultado['b1_estimado']  = historico_b1
df_resultado['y_predito']    = historico_y_pred
df_resultado['residuo']      = historico_residuo
df_resultado.to_csv('dados_rls.csv', index=False)
print("[OK] Arquivo 'dados_rls.csv' salvo.")

# =============================================================================
# 7. GRAFICOS
# =============================================================================

fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False)
fig.suptitle('Script 2: Identificacao Recursiva (RLS) - Modelo ARX\n'
             'Rede 5G/OTN: Estimacao de Parametros e Analise de Residuos',
             fontsize=13, fontweight='bold')

# --- Grafico 1: Predicao vs Real ---
ax1 = axes[0]
ax1.plot(t, y,              color='steelblue', linewidth=0.6, alpha=0.7,
         label='Atraso real y[k] (medido)')
ax1.plot(t, historico_y_pred, color='darkorange', linewidth=1.2, alpha=0.9,
         label='Predicao RLS y_hat[k]')
ax1.axvline(INSTANTE_FALHA, color='red', linestyle=':', linewidth=1.5,
            label='Falha no enlace OTN')
ax1.axvspan(INSTANTE_FALHA, t[-1], alpha=0.05, color='red')
ax1.set_ylabel('Atraso [ms]')
ax1.set_xlabel('Tempo [s]')
ax1.set_title('Atraso Real vs Predicao do Modelo ARX')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.4)

# --- Grafico 2: Evolucao dos parametros a1 e b1 ---
ax2 = axes[1]
ax2_b = ax2.twinx()
ax2.plot(t, historico_a1,  color='purple', linewidth=1.2,
         label='a1 (memoria da fila) [eixo esq]')
ax2_b.plot(t, historico_b1, color='teal', linewidth=1.2, linestyle='--',
           label='b1 (impacto do trafego) [eixo dir]')
ax2.axvline(INSTANTE_FALHA, color='red', linestyle=':', linewidth=1.5)
ax2.axvspan(INSTANTE_FALHA, t[-1], alpha=0.05, color='red')
ax2.set_ylabel('a1', color='purple')
ax2_b.set_ylabel('b1', color='teal')
ax2.set_xlabel('Tempo [s]')
ax2.set_title('Evolucao dos Parametros Estimados pelo RLS\n'
              '(Mudanca abrupta em t=30s indica a falha)')
ax2.legend(loc='upper left',  fontsize=8)
ax2_b.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.4)

# --- Grafico 3: Residuo r[k] ---
ax3 = axes[2]
ax3.plot(t, historico_residuo, color='gray', linewidth=0.6, alpha=0.8,
         label='Residuo r[k] = y[k] - y_hat[k]')
ax3.axhline(0,  color='black', linewidth=0.8)
ax3.axvline(INSTANTE_FALHA, color='red', linestyle=':', linewidth=1.5,
            label='Falha no enlace OTN')
ax3.axvspan(INSTANTE_FALHA, t[-1], alpha=0.05, color='red')
ax3.set_ylabel('Residuo [ms]')
ax3.set_xlabel('Tempo [s]')
ax3.set_title('Residuo do Modelo (Base para Deteccao de Falhas no Script 3)')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.4)

# --- Grafico 4: Autocorrelacao dos residuos (Teste de Brancura) ---
ax4 = axes[3]
centro = len(residuo_estavel) - 1
lag_range = 50
lags_plot  = lags[centro:centro+lag_range+1]
autocorr_plot = autocorr[centro:centro+lag_range+1]
ax4.stem(lags_plot, autocorr_plot, linefmt='gray', markerfmt='o',
         basefmt='k-')
ax4.axhline( limite_95, color='red', linestyle='--', linewidth=1.2,
             label=f'Limite 95% (+/- {limite_95:.3f})')
ax4.axhline(-limite_95, color='red', linestyle='--', linewidth=1.2)
ax4.set_ylabel('Autocorrelacao')
ax4.set_xlabel('Defasagem (lags)')
ax4.set_title('Teste de Brancura do Residuo (periodo estavel: t < 30s)\n'
              'Picos dentro dos limites -> ARX suficiente | Fora -> usar ARMAX')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.4)
ax4.set_xlim(0, lag_range)

plt.tight_layout()
plt.savefig('grafico_rls.png', dpi=150, bbox_inches='tight')
print("[OK] Grafico salvo em 'grafico_rls.png'")
