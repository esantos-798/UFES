clear all;
clc;

warning('off', 'Octave:graphics-toolkit-deprecated');
set(0, 'defaultfigurevisible', 'off');
pkg load control;

%%PARAMETROS
dados = [0.4; 0.05; 2.5; 10.0; 0.4; 0.5];% dados gerados pelo código "dadosmotor.p" com I=5 executado separadamente:
Ts = 0.01; % período de amostragem
delta=1; %período de amostragem
tf=100; % tempo final
C3 = [0 0 1]; % matriz de saída C3 para isolar x3 (a velocidade angular)
M = 15; % numero de amostras que cobre o tempo de acomodacao do motor.Como delta=1s aqui, e o motor estabiliza em poucos segundos (ver item 1.1), M=15 amostras (15s) e mais que suficiente para capturar a dinamica.

[A, B, C, D, Ad, Bd, Cd, Dd] = gerador_matrizes(dados, Ts);
SYSC_3 = ss(A, B, C3, D);

[sbpa, t] = gerador_sequencia_binaria_aleatoria(delta, tf);
u_prbs = sbpa(:);
N = length(u_prbs);
N_half = floor(N/2) + 1;

%%% ---- Simulacao da planta (saida velocidade, SYSC_3) com o PRBS ----
y_sem_ruido = lsim(SYSC_3, u_prbs, t);
y_sem_ruido = y_sem_ruido(:);

% Adiciona ruido branco gaussiano media zero, desvio padrao 1e-2
sigma_ruido = 1e-2;
y_com_ruido = y_sem_ruido + sigma_ruido * randn(size(y_sem_ruido));

%%% ---- Estimacao da resposta ao impulso por minimos quadrados ----
h_est_sem_ruido = toeplitz(M, N, u_prbs, y_sem_ruido);
h_est_com_ruido = toeplitz(M, N, u_prbs, y_com_ruido);

%%% ---- Resposta ao impulso "verdadeira" do motor (continua) ----
t_h = (0:delta:M*delta)';
h_true = impulse(SYSC_3, t_h);

%%% ---- Estimacao da resposta ao impulso por Transformada de Fourier ----
H_est_sem_ruido = fourier(u_prbs, y_sem_ruido);
H_est_com_ruido = fourier(u_prbs, y_com_ruido);



%%% ---- Resposta em frequência real do motor (discreto) ----
%% Vetor de frequencias correspondente (em rad/s)
fs = 1/delta;
f = (0:N-1)*(fs/N);
w = 2*pi*f;
%% Resposta em frequência real do motor
SYSC_3_min = minreal(SYSC_3);
H_true = freqresp(SYSC_3_min, w);
H_true = squeeze(H_true);

%%% ---- Comparacao grafica ----
fig = figure('visible', 'off');
graphics_toolkit(fig, 'gnuplot');
 
stem(t_h, h_true, 'g--x', 'LineWidth', 1.8); hold on;
stem(t_h, h_est_sem_ruido, 'b--o', 'LineWidth', 1.2);
stem(t_h, h_est_com_ruido, 'r:.', 'LineWidth', 1.2);
title('Resposta ao Impulso da Velocidade - Metodo da Convolucao (MQ)');
xlabel('Tempo (s)'); ylabel('h(t)');
grid on;

xlim([0 M*delta]);
ylim([0 0.10]);

% Posicoes verticais dos 3 itens da legenda
y1 = 0.090; y2 = 0.082; y3 = 0.074;
x_marker = 8.3;   % posicao X do marcador (um pouco a esquerda do texto)
x_texto  = 9;     % posicao X do texto

% Marcadores (mesmo estilo das curvas, so o simbolo, sem linha)
plot(x_marker, y1, 'gx', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(x_marker, y2, 'bo', 'MarkerSize', 8, 'LineWidth', 1.2);
plot(x_marker, y3, 'r.', 'MarkerSize', 14);

% Textos correspondentes
text(x_texto, y1, 'Impulso real do motor', 'color', 'g');
text(x_texto, y2, 'Estimado (sem ruido)', 'color', 'b');
text(x_texto, y3, 'Estimado (com ruido)', 'color', 'r');

drawnow;
print(fig, 'resposta_impulso_questao2_1.png', '-dpng');
disp('Grafico gerado com sucesso: resposta_impulso_questao2_1.png');
 
%%% ---- Metricas de erro ----
erro_sem_ruido = norm(h_true - h_est_sem_ruido) / norm(h_true);
erro_com_ruido = norm(h_true - h_est_com_ruido) / norm(h_true);
printf('Erro relativo (sem ruido): %.4f\n', erro_sem_ruido);
printf('Erro relativo (com ruido): %.4f\n', erro_com_ruido);

fig2 = figure('visible', 'off');
graphics_toolkit(fig2, 'gnuplot');
stem(f(1:N_half), abs(H_true(1:N_half)), 'g--x', 'LineWidth', 1.8); hold on;
stem(f(1:N_half), abs(H_est_sem_ruido(1:N_half)), 'b--o', 'LineWidth', 1.2);
stem(f(1:N_half), abs(H_est_com_ruido(1:N_half)), 'r:.', 'LineWidth', 1.2);
title('Resposta ao Impulso da Velocidade - Transformada de Fourier ');
xlabel('Frequência (Hz)'); ylabel('H(f)');
grid on;

xlim([0 f(N_half)]);   % ate a frequencia de Nyquist real
ylim([0 0.45]);

% Posicoes verticais dos 3 itens da legenda
y1 = 0.42; y2 = 0.38; y3 = 0.34;
x_marker = f(N_half)*0.56;   % posicao X do marcador (um pouco a esquerda do texto)
x_texto  = f(N_half)*0.6;     % posicao X do texto

% Marcadores (mesmo estilo das curvas, so o simbolo, sem linha)
plot(x_marker, y1, 'gx', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(x_marker, y2, 'bo', 'MarkerSize', 8, 'LineWidth', 1.2);
plot(x_marker, y3, 'r.', 'MarkerSize', 14);

% Textos correspondentes
text(x_texto, y1, 'Impulso real do motor', 'color', 'g');
text(x_texto, y2, 'Estimado (sem ruido)', 'color', 'b');
text(x_texto, y3, 'Estimado (com ruido)', 'color', 'r');

drawnow;
print(fig2, 'resposta_impulso_questao2_2.png', '-dpng');
disp('Grafico gerado com sucesso: resposta_impulso_questao2_2.png');

fig3 = figure('visible', 'off');
graphics_toolkit(fig3, 'gnuplot');
stem(f, H_true, 'g--x', 'LineWidth', 1.8); hold on;
stem(f, H_est_sem_ruido, 'b--o', 'LineWidth', 1.2);
stem(f, H_est_com_ruido, 'r:.', 'LineWidth', 1.2);
title('Resposta ao Impulso da Velocidade - Transformada de Fourier ');
xlabel('Frequência (Hz)'); ylabel('H(f)');
grid on;

% Legenda manual (ajuste as coordenadas conforme a escala real deste grafico)
text(10, 0.35, 'Impulso real do motor', 'color', 'g');
text(10, 0.30, 'Estimado (sem ruido)', 'color', 'b');
text(10, 0.25, 'Estimado (com ruido)', 'color', 'r');

drawnow;
print(fig3, 'resposta_impulso_questao2_2_1.png', '-dpng');
disp('Grafico gerado com sucesso: resposta_impulso_questao2_2_1.png');
 
%%% ---- Metricas de erro  ----
Erro_sem_ruido = norm(H_true(1:N_half) - H_est_sem_ruido(1:N_half)) / norm(H_true(1:N_half));
Erro_com_ruido = norm(H_true(1:N_half) - H_est_com_ruido(1:N_half)) / norm(H_true(1:N_half));
printf('Erro relativo (sem ruido): %.4f\n', Erro_sem_ruido);
printf('Erro relativo (com ruido): %.4f\n', Erro_com_ruido);

