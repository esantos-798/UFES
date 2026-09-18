clear all;
clc;
% Suprime os avisos de depreciação dos toolkits gráficos no CLI
warning('off', 'Octave:graphics-toolkit-deprecated');

% Força a não exibição de janelas
set(0, 'defaultfigurevisible', 'off');

% Carrega o pacote de sistemas de controle
pkg load control;

%%PARAMETROS
dados = [0.4; 0.05; 2.5; 10.0; 0.4; 0.5];% dados gerados pelo código "dadosmotor.p" com I=5 executado separadamente ("Dados Hardcoded")
Ts = 0.01; % período de amostragem
C3 = [0 0 1]; % matriz de saída C3 para isolar x3 (a velocidade angular)
C2 = [0 1 0]; % matriz de saída C2 para isolar x2 (posição angular)

[A, B, C, D, Ad, Bd, Cd, Dd] = gerador_matrizes(dados, Ts)
%%%% ITEM 1.1 - Simulação ao Degrau de 10 V e FTs Contínuas
%%% Cria um vetor de tempo de 0 a 10 segundos com passo Ts=0.01
t_sim = 0:Ts:10;
%%% Gera o sinal de entrada u(t)=10V constante (um degrau de tensão de 10V)
u_10v = 10 * ones(size(t_sim));
n = length(u_10v);

%%% A equação geral de saída no espaço de estados é dada por "y(t)=C.x(t)+D.u(t)". Portanto:
%% FT3(s) = X3(s)/U(s) - Velocidade

% Cria o sistema referente a saída C3 em espaço de estados contínuo SYSC_3.
SYSC_3 = ss(A, B, C3, D);
% Converte a representação em espaço de estados G3(s) na Função de Transferência contínua que relaciona a Tensão de Entrada U(s) com a Velocidade Angular X3(s)
G3_s = tf(SYSC_3);

%% FT2(s) = X2(s)/U(s) - Posição
% Cria o sistema referente a saída C2 em espaço de estados contínuo SYSC_2.
SYSC_2 = ss(A, B, C2, D);
% Converte a representação em espaço de estados G2(s) na Função de Transferência contínua que relaciona a Tensão de Entrada U(s) com a Posição Angular X2(s)
G2_s = tf(SYSC_2);


%% ITEM 1.2 - FTs Discretas e Polinômios ARX
% i) G3(z) - Velocidade
G3_z = c2d(G3_s, Ts, 'zoh');

% ii) G2(z) - Posição
G2_z = c2d(G2_s, Ts, 'zoh');

% iii) Polinômios ARX A(z) e B(z) para Velocidade
[num_g3, den_g3] = tfdata(G3_z, 'v');
[num_g2, den_g2] = tfdata(G2_z, 'v');

disp('=== ITEM 1.2 ===');
disp('Polinômio A(z) da Velocidade:'); disp(den_g3);
disp('Polinômio B(z) da Velocidade:'); disp(num_g3);
disp('Polinômio A(z) da Posição:'); disp(den_g2);
disp('Polinômio B(z) da Posição:'); disp(num_g2);

%% ITEM 1.3 - Comparação Gráfica
[y_cont, t_out] = lsim(SYSC_3, u_10v, t_sim);
[y_disc, ~]     = lsim(G3_z, u_10v, t_sim);
fig = figure('visible', 'off');
graphics_toolkit(fig, 'gnuplot');
plot(t_out, y_cont, 'b-', 'LineWidth', 2); hold on;
plot(t_out, y_disc, 'r--', 'LineWidth', 1.5);
title('Comparacao da Resposta ao Degrau (10V) - Velocidade Angular');
xlabel('Tempo (s)'); ylabel('Velocidade (rad/s)');
grid on;

xlim([0 10]);
ylim([0 4]);

% Posicoes da legenda manual
y1 = 1.0; y2 = 0.7;
x_texto = 7.0;

% Textos correspondentes
text(x_texto, y1, 'Continuo G_3(s)', 'color', 'b');
text(x_texto, y2, 'Discreto G_3(z)', 'color', 'r');

drawnow;
print(fig, 'resposta_degrau_questao1.png', '-dpng');
disp('Gráfico gerado com sucesso: resposta_degrau_questao1.png');