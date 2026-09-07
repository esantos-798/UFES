% =========================================================
%  Simulação do modelo de LED — com codificação 8B10B
%  e Diagrama de Olho
% =========================================================
%                                         Cristiano Tavares
%                                                Maio/2026
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

% ── Parâmetros do modelo ──────────────────────────────────
Lum_max  = 755;         % Iluminância máxima  [lux]
zeta     = 20;          % Parâmetro de saturação  (adimensional, > 0)
k        = 1.9;         % Fator de forma (inteiro positivo)
SNR_dB   = 30;          % Relação sinal-ruído
R        = 0.6;         % Responsatividade (Camporez, et al. 2024)
N_bytes  = 1000;        % Número de bytes (cada byte → 10 bits após 8B10B)
I_bias   = 800;         % Corrente de polarização [mA]
I_mod_A  = 10;          % Amplitude da corrente modulante [mA]
alpha    = 10;
Tempo_pulso = 4;        % Amostras por bit (inteiro)


% ============================================================
%  CURVA DE LUMINÂNCIA
% ============================================================

I_LED = linspace(0, 2500, 500);
Lum   = I_LED ./ (zeta + (I_LED / Lum_max).^(2*k)).^(1/(2*k));

figure(1);
plot(I_LED, Lum, 'LineWidth', 2);
grid on;
xlabel('Corrente I_{LED} [mA]', 'FontSize', 12);
ylabel('Luminância Lum [lux]',  'FontSize', 12);
title('Curva de Luminância do LED', 'FontSize', 14);
xlim([0 2500]);
ylim([0 Lum_max * 1.1]);


% =========================================================
%  GERAÇÃO DOS BITS  →  CODIFICAÇÃO 8B10B
% =========================================================

rng(42);                                        % Reprodutibilidade
bytes_seq = randi([0 1], 1, N_bytes * 8);       % Bits aleatórios (múltiplo de 8)

% Codificação 8B10B — retorna vetor de bits 8B10B
bits_8b10b = enc8b10b(bytes_seq);               % Comprimento: N_bytes * 10

N_bits = length(bits_8b10b);                    % Total de bits codificados
N_uso  = N_bits * Tempo_pulso;                  % Total de amostras


% =========================================================
%  MODULAÇÃO DO LED  —  I_fin = I_bias + alpha * I_mod
% =========================================================

% 1) NRZ bipolar a partir dos bits 8B10B
niveis = (2 * bits_8b10b - 1) * I_mod_A;       % +I_mod_A ou -I_mod_A
I_mod  = rectpulse(niveis, Tempo_pulso);

% 2) Sinal de corrente final
I_fin = I_bias + alpha * I_mod;

% 3) Luminância de saída
Lum_fin = I_fin ./ (zeta + (I_fin / Lum_max).^(2*k)).^(1/(2*k));

% 4) Fotodetecção
Ir = R * abs(Lum_fin).^2;                       % (Pereira, et al. 2015)

% 5) DC Block
Irb = Ir - mean(Ir);

% 6) Ruído no canal (compensado pelo ganho de integração)
SNR_ch = SNR_dB - 10*log10(Tempo_pulso) + 3;
Ir_ruido = awgn(Irb, SNR_ch, 'measured');


% =========================================================
%  VISUALIZAÇÃO DO SINAL MODULADO (primeiros 10 bits)
% =========================================================

t = 1:N_uso;
n_mostrar = 10 * Tempo_pulso;   % Janela de 10 bits

figure(2);
subplot(2,1,1);
plot(t(1:n_mostrar), I_fin(1:n_mostrar), 'LineWidth', 2);
grid on;
title('Sinal de Corrente Modulado 8B10B (primeiros 10 bits codificados)');
ylabel('I_{fin} [mA]');
xlim([1 n_mostrar]);
ylim([min(I_fin)-10, max(I_fin)+10]);

subplot(2,1,2);
plot(t(1:n_mostrar), Lum_fin(1:n_mostrar), 'LineWidth', 1.5);
grid on;
title('Luminância de Saída (Lum_{fin})');
ylabel('Luminância [lux]');
xlabel('Amostras');
xlim([1 n_mostrar]);
ylim([min(Lum_fin)-5, max(Lum_fin)+5]);

sgtitle('Modulação NRZ Bipolar com codificação 8B10B', 'FontSize', 14);


% =========================================================
%  DIAGRAMA DE OLHO
% =========================================================
% Abre janela separada para cada diagrama — sem ruído e com ruído.
% eyediagram(sinal, spp) traça o olho com 'spp' amostras por símbolo,
% varrendo o sinal inteiro e sobrepondo todas as janelas de 2 bits.
% Com 8B10B e NRZ bipolar o período de símbolo = Tempo_pulso amostras.

spp = Tempo_pulso;  % Amostras por bit (símbolo NRZ)

figure(3);
eyediagram(Irb, 2*spp);
title(['Diagrama de Olho — Sem Ruído (SNR = \infty)'], 'FontSize', 13);
xlabel('Tempo (amostras)');
ylabel('Amplitude');

figure(4);
eyediagram(Ir_ruido, 2*spp);
title(['Diagrama de Olho — Com Ruído (SNR_{canal} = ', ...
       num2str(SNR_dB, '%d'), ' dB)'], 'FontSize', 13);
xlabel('Tempo (amostras)');
ylabel('Amplitude');


% ============================================================
%  REFERÊNCIAS
% ============================================================
% H. Camporez et al., "AI-Driven Enhancements for Handover in Visible Light
%   Communication Systems," J. Lightwave Technol., vol. 42, no. 23,
%   pp. 8191-8202, Dec. 2024, doi: 10.1109/JLT.2024.3430188.
% E. d. V. Pereira et al., "Impact of Optical Power in the Guard-Band
%   Reduction of an Optimized DDO-OFDM System," J. Lightwave Technol.,
%   vol. 33, no. 23, pp. 4717-4725, Dec. 2015, doi: 10.1109/JLT.2015.2481085.
% PROAKIS, J. G.; SALEHI, M. Digital Communications. 5. ed. McGraw-Hill, 2008.
