%% ========================================================================
%  Questao 2.6 - Identificacao do motor CC (I=5) pelo METODO DA
%  AUTOCORRELACAO, usando o mesmo sinal PRBS e os mesmos parametros do
%  motor ja usados na Lista 1.1 (lista1_1_item2.m).
%
%  2.6.1 - Resposta ao impulso via autocorrelacao (Aguirre, Ex. 4.2.6)
%  2.6.2 - Resposta em frequencia via densidade espectral (Aguirre, Ex. 4.4.1)
%
%  Formulas usadas (das notas do professor - "Identificacao da Resposta
%  ao Impulso atraves de Funcao de Correlacao"):
%
%     r_uy(k) = (1/N) * sum_i u(i) y(i+k)         (correlacao cruzada)
%     r_uu(k) = (1/N) * sum_i u(i) u(i+k)         (autocorrelacao de u)
%
%     Equacao de Wiener-Hopf: r_uy(k) = sum_i h(i) r_uu(k-i)
%
%     Como o PRBS (sinal pseudo-aleatorio +-1) tem r_uu(k) ~ 0 para k~=0
%     ("sinal pseudo-aleatorio"), a matriz R_uu fica ~diagonal e a
%     equacao se reduz a:   h(k) = r_uy(k) / r_uu(0)
%
%     Em frequencia (Teorema de Wiener-Khinchin): S_uy(w) = FFT(r_uy),
%     S_uu(w) = FFT(r_uu), e H(w) = S_uy(w)/S_uu(w).
%
%  Requer no mesmo diretorio: gerador_matrizes.m, gerador_sequencia_
%  binaria_aleatoria.m (ambos ja usados na Lista 1.1) e o pacote
%  'control' do Octave (pkg load control).
% ========================================================================
clear all; close all; clc;

if exist('OCTAVE_VERSION', 'builtin') ~= 0
    warning('off', 'Octave:graphics-toolkit-deprecated');
    pkg load control;
end

%% --- 0) Parametros do motor (I = 5), identicos a Lista 1.1 ----------------
dados = [0.4; 0.05; 2.5; 10.0; 0.4; 0.5];  % [Ra; La; Kb; Jm; K; B], de dadosmotor.p (I=5)
Ts = 0.01;
C3 = [0 0 1];     % saida = velocidade angular (x3)

[A, B, C, D] = gerador_matrizes(dados, Ts);
SYSC_3 = ss(A, B, C3, D);

%% --- 1) Sinal PRBS: MESMO gerador da Lista 1.1 -----------------------------
% IMPORTANTE: gerador_sequencia_binaria_aleatoria usa rand() sem semente
% fixa, entao cada execucao gera um PRBS DIFERENTE. Fixamos a semente
% aqui para que o resultado seja reprodutivel e comparavel ao da Lista 1.1
% (ajuste/remova a semente se quiser comparar com uma execucao especifica
% ja salva).
if exist('OCTAVE_VERSION', 'builtin') ~= 0
    rand('seed', 5);
else
    rng(5);
end

delta = 1; tf = 100;
M = 15;   % mesma janela de lags usada na Lista 1.1 (cobre o regime transitorio)

[sbpa, t] = gerador_sequencia_binaria_aleatoria(delta, tf);
u = sbpa(:);
N = length(u);

%% --- 2) Simulacao da planta (com e sem ruido) ------------------------------
y_sem_ruido = lsim(SYSC_3, u, t);
y_sem_ruido = y_sem_ruido(:);

sigma_ruido = 1e-2;
y_com_ruido = y_sem_ruido + sigma_ruido * randn(size(y_sem_ruido));

%% ========================================================================
%  2.6.1 - RESPOSTA AO IMPULSO PELO METODO DA AUTOCORRELACAO
%  ========================================================================

ubar = mean(u);
r_uu = zeros(M+1, 1);
for k = 0:M
    idx = 1:(N-k);
    r_uu(k+1) = sum( (u(idx)-ubar) .* (u(idx+k)-ubar) ) / N;
end

fprintf('=== Verificacao: r_uu(k) para k=0..5 (deve ser ~0 para k~=0) ===\n');
disp(r_uu(1:6)');

calcula_ruy = @(y) arrayfun(@(k) sum((u(1:(N-k))-ubar).*(y(1+k:N)-mean(y))) / N, 0:M)';

r_uy_sem = calcula_ruy(y_sem_ruido);
r_uy_com = calcula_ruy(y_com_ruido);

% Como r_uu(k) ~ 0 para k~=0 (verificado acima), aplicamos a formula
% simplificada h(k) = r_uy(k) / r_uu(0):
h_est_sem_ruido = r_uy_sem / r_uu(1);   % r_uu(1) = r_uu(k=0)
h_est_com_ruido = r_uy_com / r_uu(1);

%% --- Resposta ao impulso "verdadeira" (continua) ---------------------------
t_h = (0:delta:M*delta)';
h_true = impulse(SYSC_3, t_h);

%% --- Grafico comparativo ----------------------------------------------------
figure('Name', 'Questao 2.6.1 - Resposta ao impulso (metodo da autocorrelacao)');
stem(t_h, h_true, 'g--x', 'LineWidth', 1.8); hold on;
stem(t_h, h_est_sem_ruido, 'b--o', 'LineWidth', 1.2);
stem(t_h, h_est_com_ruido, 'r:.', 'LineWidth', 1.2);
hold off;
grid on;
xlabel('Tempo (s)'); ylabel('h(t)');
title('Resposta ao Impulso - Metodo da Autocorrelacao', 'Interpreter', 'none');
legend('Impulso real', 'Estimado (sem ruido)', 'Estimado (com ruido)', 'Location', 'best');

drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'resposta_impulso_autocorrelacao_2_6_1.png');
fprintf('\nFigura salva: resposta_impulso_autocorrelacao_2_6_1.png\n');

erro_sem_ruido = norm(h_true - h_est_sem_ruido) / norm(h_true);
erro_com_ruido = norm(h_true - h_est_com_ruido) / norm(h_true);
fprintf('\n=== 2.6.1 - Erros relativos (metodo da AUTOCORRELACAO) ===\n');
fprintf('Erro relativo (sem ruido): %.4f\n', erro_sem_ruido);
fprintf('Erro relativo (com ruido): %.4f\n', erro_com_ruido);

%% ========================================================================
%  2.6.2 - RESPOSTA EM FREQUENCIA PELA DENSIDADE ESPECTRAL (AUTOCORRELACAO)
%  ========================================================================

% Densidades espectrais via FFT das funcoes de correlacao (Wiener-Khinchin).
% Para comparar com o metodo de Fourier direto (Lista 1.1), usamos o MESMO
% grid de frequencias f = (0:N-1)*(fs/N).
r_uu_full = zeros(N,1);
r_uy_sem_full = zeros(N,1);
r_uy_com_full = zeros(N,1);
for k = 0:(N-1)
    idx = 1:(N-k);
    r_uu_full(k+1)      = sum((u(idx)-ubar).*(u(idx+k)-ubar)) / N;
    r_uy_sem_full(k+1)  = sum((u(idx)-ubar).*(y_sem_ruido(idx+k)-mean(y_sem_ruido))) / N;
    r_uy_com_full(k+1)  = sum((u(idx)-ubar).*(y_com_ruido(idx+k)-mean(y_com_ruido))) / N;
end

S_uu     = fft(r_uu_full);
S_uy_sem = fft(r_uy_sem_full);
S_uy_com = fft(r_uy_com_full);

H_est_sem_ruido = S_uy_sem ./ S_uu;
H_est_com_ruido = S_uy_com ./ S_uu;

%% --- Resposta em frequencia "verdadeira" (discreta) ------------------------
fs = 1/delta;
f  = (0:N-1)*(fs/N);
w  = 2*pi*f;
N_half = floor(N/2) + 1;

SYSC_3_min = minreal(SYSC_3);
H_true = squeeze(freqresp(SYSC_3_min, w));

%% --- Grafico comparativo (modulo) -------------------------------------------
figure('Name', 'Questao 2.6.2 - Resposta em frequencia (metodo da autocorrelacao)');
stem(f(1:N_half), abs(H_true(1:N_half)), 'g--x', 'LineWidth', 1.8); hold on;
stem(f(1:N_half), abs(H_est_sem_ruido(1:N_half)), 'b--o', 'LineWidth', 1.2);
stem(f(1:N_half), abs(H_est_com_ruido(1:N_half)), 'r:.', 'LineWidth', 1.2);
hold off;
grid on;
xlabel('Frequencia (Hz)'); ylabel('|H(f)|');
title('Resposta em Frequencia - Metodo da Autocorrelacao/PSD', 'Interpreter', 'none');
legend('Real', 'Estimado (sem ruido)', 'Estimado (com ruido)', 'Location', 'best');

drawnow;
quadro2 = getframe(gcf);
imwrite(quadro2.cdata, 'resposta_frequencia_autocorrelacao_2_6_2.png');
fprintf('\nFigura salva: resposta_frequencia_autocorrelacao_2_6_2.png\n');

Erro_sem_ruido = norm(H_true(1:N_half) - H_est_sem_ruido(1:N_half)) / norm(H_true(1:N_half));
Erro_com_ruido = norm(H_true(1:N_half) - H_est_com_ruido(1:N_half)) / norm(H_true(1:N_half));
fprintf('\n=== 2.6.2 - Erros relativos (metodo da AUTOCORRELACAO) ===\n');
fprintf('Erro relativo (sem ruido): %.4f\n', Erro_sem_ruido);
fprintf('Erro relativo (com ruido): %.4f\n', Erro_com_ruido);

fprintf(['\n--- Para a discussao pedida (convolucao vs autocorrelacao) ---\n' ...
         'Compare estes erros com os obtidos no lista1_1_item2.m (metodo da\n' ...
         'convolucao/MQ via Toeplitz, e metodo de Fourier direto). A\n' ...
         'expectativa teorica (das notas do professor: "as funcoes de\n' ...
         'correlacao FCC e FAC sao relativamente robustas a ruido devido ao\n' ...
         'efeito de media") e que o metodo da autocorrelacao degrade MENOS\n' ...
         'com ruido do que a inversao direta via Toeplitz (que pode\n' ...
         'amplificar ruido se a matriz U for mal-condicionada).\n']);