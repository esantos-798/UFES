"""
==============================================================================
PROJETO - Identificação de Sistemas | UFES 2026
Tema: Predição de Atraso e Detecção de Falhas em Redes 5G/OTN

SCRIPT 1 de 3: Simulação da Fila de Rede (Geração de Dados)
==============================================================================

CONTEXTO:
    Simulamos um nó de transporte (OTN) no enlace Fronthaul de uma rede 5G.
    Pacotes de serviços URLLC e eMBB chegam ao nó e formam uma fila.
    O atraso de enfileiramento (Queueing Delay) é nossa variável de interesse.

    A "FALHA" acontece em t=INSTANTE_FALHA segundos: a capacidade do enlace
    cai abruptamente (simula degradação física do link óptico OTN), fazendo
    a fila crescer e o atraso disparar.

ENTRADAS E SAÍDAS DO SISTEMA (para Identificação de Sistemas):
    u[k] = taxa de chegada de tráfego no instante k  [Mbps]  -> ENTRADA
    y[k] = atraso de enfileiramento no instante k    [ms]    -> SAÍDA

DEPENDÊNCIAS:
    pip install numpy matplotlib pandas
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# PARÂMETROS DA SIMULAÇÃO (Você pode alterar esses valores para experimentar)
# =============================================================================

# --- Tempo ---
TS          = 0.01          # Período de amostragem: 10 ms (polling da gerência)
T_TOTAL     = 60.0          # Duração total da simulação: 60 segundos
INSTANTE_FALHA = 30.0       # Instante em que ocorre a falha no enlace OTN [s]

# --- Rede OTN (o nosso "sistema") ---
C_NORMAL    = 10.0          # Capacidade do enlace em operação normal   [Gbps]
C_FALHA     = 3.0           # Capacidade do enlace após a falha         [Gbps]

# --- Tráfego 5G (entrada u[k]) ---
TRAFEGO_MEDIO  = 6.0        # Taxa média de tráfego chegando             [Gbps]
TRAFEGO_STD    = 1.5        # Desvio padrão (rafadas/variações normais)  [Gbps]

# --- Ruído de Medição ---
# Simula imprecisões na leitura via NETCONF / gerência da rede
RUIDO_STD   = 0.05          # Desvio padrão do ruído de medição          [ms]

# --- Semente aleatória (garante reprodutibilidade dos resultados) ---
SEMENTE     = 42
np.random.seed(SEMENTE)

# =============================================================================
# GERAÇÃO DO VETOR DE TEMPO E TRÁFEGO (Entrada u[k])
# =============================================================================

t = np.arange(0, T_TOTAL, TS)   # Vetor de tempo discreto
N = len(t)                        # Número total de amostras

# Tráfego chegando: processo aleatório Gaussiano (simula variações reais do 5G)
# Limitamos entre 0 e 1.2 * C_NORMAL para não gerar valores absurdos
u = np.random.normal(TRAFEGO_MEDIO, TRAFEGO_STD, N)
u = np.clip(u, 0, 1.2 * C_NORMAL)

# =============================================================================
# SIMULAÇÃO DA FILA (Modelo de Fila Discreta - D/D/1 Simplificada)
# =============================================================================
# Ideia: O buffer acumula tráfego que não consegue ser transmitido.
#        A cada instante k, se u[k] > C, o excesso vai para o buffer.
#        O atraso é proporcional ao tamanho do buffer dividido pela capacidade.
#
# Isso traduz diretamente nossa equação ARX:
#   y[k] = a1 * y[k-1] + b1 * u[k-1] + e[k]
#
# onde a1 captura a "memória" da fila (quanto do buffer anterior persiste)
# e b1 captura o quanto o tráfego atual impacta no atraso.

# Vetor de capacidade (muda no instante da falha)
C = np.where(t < INSTANTE_FALHA, C_NORMAL, C_FALHA)

# Simulação iterativa da fila
buffer     = np.zeros(N)    # Tamanho do buffer em cada instante [Gb acumulado]
y_ideal    = np.zeros(N)    # Atraso sem ruído de medição        [ms]

FATOR_MEMORIA = 0.85        # Quanto do buffer persiste ao próximo instante
                            # (modela o esvaziamento gradual da fila)

for k in range(1, N):
    excesso      = max(0, u[k] - C[k])        # Tráfego que não coube no enlace
    buffer[k]    = FATOR_MEMORIA * buffer[k-1] + excesso * TS
    buffer[k]    = max(0, buffer[k])           # Buffer não pode ser negativo
    y_ideal[k]   = (buffer[k] / C[k]) * 1000  # Converte para milissegundos [ms]

# Adiciona ruído de medição (imprecisão da gerência NETCONF)
ruido   = np.random.normal(0, RUIDO_STD, N)
y       = np.abs(y_ideal + ruido)             # Atraso observado [ms] (sempre >= 0)

# =============================================================================
# SALVAR OS DADOS EM CSV (serão lidos pelo Script 2)
# =============================================================================

df = pd.DataFrame({
    'k':            np.arange(N),      # Índice da amostra
    'tempo_s':      t,                 # Tempo em segundos
    'trafego_u':    u,                 # Entrada:  tráfego chegando [Gbps]
    'capacidade_C': C,                 # Capacidade do enlace [Gbps] (referência)
    'atraso_y':     y,                 # Saída: atraso medido [ms]
    'atraso_ideal': y_ideal            # Atraso sem ruído (só para comparação)
})

df.to_csv('dados_simulacao.csv', index=False)
print("[OK] Arquivo 'dados_simulacao.csv' salvo com sucesso!")
print(f"   Total de amostras: {N}")
print(f"   Duracao: {T_TOTAL}s | Ts: {TS*1000:.0f}ms")
print(f"   Falha simulada em: t = {INSTANTE_FALHA}s")
print(f"   Capacidade: {C_NORMAL} Gbps -> {C_FALHA} Gbps apos a falha")

# =============================================================================
# GRÁFICOS
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle('Script 1: Simulação da Fila de Rede 5G/OTN\n'
             '(Dados para Identificação de Sistemas)', fontsize=13, fontweight='bold')

# --- Gráfico 1: Tráfego de Entrada u[k] ---
ax1 = axes[0]
ax1.plot(t, u, color='steelblue', linewidth=0.8, alpha=0.8, label='Tráfego u[k] [Gbps]')
ax1.axhline(C_NORMAL, color='green',  linestyle='--', linewidth=1.2, label=f'Capacidade Normal ({C_NORMAL} Gbps)')
ax1.axhline(C_FALHA,  color='orange', linestyle='--', linewidth=1.2, label=f'Capacidade Pós-Falha ({C_FALHA} Gbps)')
ax1.axvline(INSTANTE_FALHA, color='red', linestyle=':', linewidth=1.5)
ax1.set_ylabel('Tráfego [Gbps]')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.4)

# --- Gráfico 2: Atraso de Fila y[k] ---
ax2 = axes[1]
ax2.plot(t, y_ideal, color='green', linewidth=1.2, alpha=0.7, label='Atraso ideal (sem ruído)')
ax2.plot(t, y,       color='steelblue', linewidth=0.7, alpha=0.7, label='Atraso medido y[k] (com ruído) [ms]')
ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Limite URLLC = 1 ms')
ax2.axvline(INSTANTE_FALHA, color='red', linestyle=':', linewidth=1.5, label='Falha no enlace OTN')
ax2.set_ylabel('Atraso [ms]')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.4)

# --- Gráfico 3: Buffer acumulado ---
ax3 = axes[2]
ax3.fill_between(t, buffer, color='salmon', alpha=0.6, label='Buffer acumulado [Gb]')
ax3.axvline(INSTANTE_FALHA, color='red', linestyle=':', linewidth=1.5)
ax3.set_ylabel('Buffer [Gb]')
ax3.set_xlabel('Tempo [s]')
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.4)

# Marca o instante da falha em todos os gráficos
for ax in axes:
    ax.axvspan(INSTANTE_FALHA, T_TOTAL, alpha=0.05, color='red')

plt.tight_layout()
plt.savefig('grafico_simulacao.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Grafico salvo em 'grafico_simulacao.png'")
