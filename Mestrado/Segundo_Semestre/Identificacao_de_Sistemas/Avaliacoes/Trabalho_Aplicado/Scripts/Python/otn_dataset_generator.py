import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parâmetros da simulação
ts = 0.2          # Tempo de amostragem (NETCONF polling a cada 200ms)
t_final = 15.0    # Duração total da simulação em segundos
t = np.arange(0, t_final, ts)

# Parâmetros Físicos do Equipamento OTN (FOPDT)
atraso_gerencia = 2.0  # Tempo morto (Dead Time) até o NETCONF responder
tau = 2.5              # Constante de tempo de convergência (sincronismo)
ganho = 100.0          # Chega a 100% de estabilidade

# Criação do Vetor de Entrada u(t) - O Comando NETCONF
instante_comando = 1.0
u = np.zeros(len(t))
u[t >= instante_comando] = 1.0

# Criação do Vetor de Saída y(t) - A Estabilidade
y = np.zeros(len(t))
for i, tempo in enumerate(t):
    tempo_efetivo = tempo - instante_comando - atraso_gerencia
    if tempo_efetivo > 0:
        # Resposta de um sistema de 1ª ordem ao degrau
        y[i] = ganho * (1 - np.exp(-tempo_efetivo / tau))

# Adicionando Ruído (Flutuações reais do FEC e polling)
ruido = np.random.normal(0, 3.0, len(t))
y_ruidoso = y + ruido
y_ruidoso[y_ruidoso < 0] = 0 # Estabilidade não pode ser menor que 0

# Salvando os dados para a próxima etapa da disciplina
df = pd.DataFrame({'Tempo_s': t, 'Comando_u': u, 'Estabilidade_y': y_ruidoso})
df.to_csv('dados_otn.csv', index=False)

# Plotando o Gráfico
plt.figure(figsize=(10, 5))
plt.plot(t, u * 100, 'r--', label='Comando NETCONF (u)')
plt.plot(t, y, 'g', linewidth=2, label='Convergência Ideal (Física)')
plt.plot(t, y_ruidoso, 'b.', label='Leitura NETCONF com Ruído (y)')
plt.title('Simulação de Provisionamento OTN (Identificação de Sistemas)')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Estabilidade (%)')
plt.legend()
plt.grid(True)
plt.show()