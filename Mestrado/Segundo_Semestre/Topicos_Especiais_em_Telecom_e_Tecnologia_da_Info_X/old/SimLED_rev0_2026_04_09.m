% =========================================================
%  Simulação do modelo de LED — Potência Fotométrica
% =========================================================
%                                         Cristiano Tavares
%                                                Abril/2026
%
%  Equação:
%                        I_LED
%  Lum_LED = ─────────────────────────────────
%            [ ζ + (I_LED / P_max)^2k ]^(1/2k)
%
%  Unidades:
%    I_LED    → corrente elétrica       [mA]
%    Lum      → luminância              [lux]
%    Lum_max  → luminância máxima       [lux]
% =========================================================

clc; clear; close all;

%% ── Parâmetros do modelo ──────────────────────────────────
Lum_max = 755;     % Iluminância máxima  [lux]
zeta  = 20;      % Parâmetro de saturação  (adimensional, > 0)
k     = 1.9;        % Fator de forma (inteiro positivo)

%% ── Corrente de entrada ───────────────────────────────────
I_LED = linspace(0, 2500, 500);   % Varredura [mA]

%% ── Função do modelo ──────────────────────────────────────
Lum = I_LED ./ (zeta + (I_LED / Lum_max).^(2*k)).^(1/(2*k));

%% ── Plotagem do Gráfico ───────────────────────────────────
figure(1); % Cria a figura com fundo branco
plot(I_LED, Lum, 'LineWidth', 2);

% Adicionando rótulos e título
grid on;
xlabel('Corrente I_{LED} [mA]', 'FontSize', 12);
ylabel('Luminância Lum [lux]', 'FontSize', 12);
title('Curva de Luminância do LED', 'FontSize', 14);

% Ajuste dos limites dos eixos para melhor visualização
xlim([0 2500]);
ylim([0 Lum_max * 1.1]);

% =========================================================
%  MODULAÇÃO DO LED  —  I_fin = I_port + I_mod
% =========================================================

% ── Parâmetros da modulação ───────────────────────────────
N_bits  = 2048;       % Número de bits da sequência
I_port = 800;         % Amplitude da coorente da portadora
I_mod_A = 50;         % Amplitude da corrente modulante [mA]
%                       bit=1 → +I_mod_A  |  bit=0 → -I_mod_A  (NRZ bipolar)
alpha = 10;

% Amostras por bit — calculado para acomodar exatamente 2048 bits
% no mesmo comprimento de i_sinal
Tempo_pulso = 5; % Amostras por bit (inteiro)
N_uso = N_bits * Tempo_pulso;               % Amostras efetivamente usadas

% 1) Sequência de bits aleatórios 
rng(42);                                    % Semente — garante reprodutibilidade
bits = randi([0 1], 1, N_bits);             % 2048 bits: 0 ou 1

% 2) Sinal modulante NRZ bipolar → corrente 
niveis = (2*bits - 1) * I_mod_A;            % +I_mod_A ou -I_mod_A
I_mod  = repelem(niveis, Tempo_pulso);      % Dá corpo ao pulso

% 3) Sinal final modulado
I_fin = I_port + alpha*I_mod;

% 4) Potência óptica de saída
Lum_fin = I_fin ./ (zeta + (I_fin / Lum_max).^(2*k)).^(1/(2*k));


% ── Plotagem da Modulação (Tempo) ─────────────────────────
figure(2);
t = 1:N_uso; % Eixo de amostras

% Subplot 1: Corrente Total Modulada (Entrada)
subplot(2,1,1);
plot(t, I_fin, 'LineWidth',2);
grid on;
title(['Sinal de Corrente Modulado (Primeiros 10 bits)']);
ylabel('I_{fin} [mA]');
% Limitando a visualização aos primeiros 10 bits (10 * Tempo_pulso)
xlim([1, 10 * Tempo_pulso]); 
ylim([min(I_fin)-10, max(I_fin)+10]);

% Subplot 2: Luminância Resultante (Saída)
subplot(2,1,2);
plot(t, Lum_fin, 'LineWidth', 1.5);
grid on;
title('Luminância de Saída (Lum_{fin})');
ylabel('Luminância [lux]');
xlabel('Amostras');
% Mesma escala de tempo para comparação
xlim([1, 10 * Tempo_pulso]); 
ylim([min(Lum_fin)-5, max(Lum_fin)+5]);

% Ajuste de espaçamento entre subplots
sgtitle('Modulação NRZ Bipolar aplicada ao LED', 'FontSize', 14);

% ── Diagrama do olho ──────────────────────────────────────
%figure(3)
eyediagram(Lum_fin,2)


% =========================================================
%  ADICIONANDO RUÍDO AO CANAL
% =========================================================
SNR_dB = 20; % Relação sinal-ruído

% Aplicando ruído na Luminância
Lum_ruido = awgn(Lum_fin, SNR_dB, 'measured');

% Plotando para comparação
figure(4);
plot(t, Lum_ruido, 'r', 'DisplayName', 'Com Ruído');
hold on;
plot(t, Lum_fin, 'b', 'LineWidth', 1.5, 'DisplayName', 'Original');
grid on;
title(['Efeito do Ruído Gaussiano (SNR = ', num2str(SNR_dB), ' dB)']);
xlabel('Amostras'); ylabel('Luminância [lux]');
legend;
xlim([1, 20 * Tempo_pulso]); % Mostrando apenas os primeiros 20 bits

% Novo Diagrama do Olho com ruído
eyediagram(Lum_ruido, 2); 
title(['Diagrama do Olho com Ruído (SNR: ', num2str(SNR_dB), 'dB)']);