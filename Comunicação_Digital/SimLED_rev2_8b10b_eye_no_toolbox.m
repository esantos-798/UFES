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
%
%  Dependências externas:
%    enc8b10b_no_toolbox.m  — deve estar no mesmo diretório
%
%  Funções do Communications Toolbox substituídas:
%    awgn        → awgn_local()       (função local)
%    rectpulse   → rectpulse_local()  (função local)
%    intdump     → intdump_local()    (função local)
%    eyediagram  → eyediagram_local() (função local)
%    sgtitle     → text() em axes     (Octave não tem sgtitle)
% =========================================================

clc; clear; close all;

% ── Parâmetros do modelo ──────────────────────────────────
Lum_max     = 755;      % Iluminância máxima  [lux]
zeta        = 20;       % Parâmetro de saturação (adimensional, > 0)
k           = 1.9;      % Fator de forma
SNR_dB      = 30;       % SNR do canal [dB]
R           = 0.6;      % Responsividade [A/W]
N_bytes     = 1000;     % Bytes de dados (→ N_bytes*10 bits após 8B10B)
I_bias      = 800;      % Corrente de polarização [mA]
I_mod_A     = 10;       % Amplitude da corrente modulante [mA]
alpha       = 10;       % Ganho de modulação
Tempo_pulso = 4;        % Amostras por bit


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

rng(42);
bytes_seq  = randi([0 1], 1, N_bytes * 8);   % bits aleatórios (múltiplo de 8)
bits_8b10b = enc8b10b_no_toolbox(bytes_seq);            % saída: N_bytes*10 bits

N_bits = length(bits_8b10b);
N_uso  = N_bits * Tempo_pulso;


% =========================================================
%  MODULAÇÃO DO LED  —  I_fin = I_bias + alpha * I_mod
% =========================================================

% 1) NRZ bipolar a partir dos bits 8B10B
niveis = (2 * bits_8b10b - 1) * I_mod_A;       % +I_mod_A ou -I_mod_A
I_mod  = rectpulse_local(niveis, Tempo_pulso);  % repete cada nível Tempo_pulso vezes

% 2) Corrente final
I_fin = I_bias + alpha * I_mod;

% 3) Luminância de saída
Lum_fin = I_fin ./ (zeta + (I_fin / Lum_max).^(2*k)).^(1/(2*k));

% 4) Fotodetecção  (Pereira et al., 2015)
Ir = R * abs(Lum_fin).^2;

% 5) DC Block
Irb = Ir - mean(Ir);

% 6) Ruído no canal (com compensação pelo ganho de integração)
SNR_ch = SNR_dB - 10*log10(Tempo_pulso) + 3;
Ir_ruido = awgn_local(Irb, SNR_ch);


% =========================================================
%  VISUALIZAÇÃO DO SINAL MODULADO (primeiros 10 bits)
% =========================================================

t = 1:N_uso;
n_mostrar = 10 * Tempo_pulso;

figure(2);
subplot(2,1,1);
plot(t(1:n_mostrar), I_fin(1:n_mostrar), 'LineWidth', 2);
grid on;
title('Corrente Modulada 8B10B (primeiros 10 bits codificados)');
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

% Título global — compatível com Octave (sem sgtitle)
axes('Position', [0 0 1 1], 'Visible', 'off');
text(0.5, 0.995, 'Modulação NRZ Bipolar com codificação 8B10B', ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 13);


% =========================================================
%  DIAGRAMA DE OLHO
% =========================================================

spp = Tempo_pulso;   % amostras por bit (símbolo NRZ)

figure(3);
eyediagram_local(Irb, 2*spp);
title('Diagrama de Olho — Sem Ruído', 'FontSize', 13);
xlabel('Amostras'); ylabel('Amplitude');

figure(4);
eyediagram_local(Ir_ruido, 2*spp);
title(['Diagrama de Olho — Com Ruído  (SNR_{canal} = ' ...
       num2str(SNR_dB, '%d') ' dB)'], 'FontSize', 13);
xlabel('Amostras'); ylabel('Amplitude');


% ============================================================
%  REFERÊNCIAS
% ============================================================
% H. Camporez et al., J. Lightwave Technol., vol. 42, no. 23, 2024.
% E. d. V. Pereira et al., J. Lightwave Technol., vol. 33, no. 23, 2015.
% PROAKIS, J. G.; SALEHI, M. Digital Communications. 5. ed. McGraw-Hill, 2008.


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FUNÇÕES LOCAIS  (sem Communications Toolbox)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---------------------------------------------------------
%  awgn_local — adiciona ruído AWGN a potência medida
%    Equivalente a: awgn(x, snr_dB, 'measured')
% ---------------------------------------------------------
function y = awgn_local(x, snr_dB)
    P_sinal = mean(x .^ 2);
    P_ruido = P_sinal / (10 ^ (snr_dB / 10));
    ruido   = sqrt(P_ruido) * randn(size(x));
    y       = x + ruido;
end

% ---------------------------------------------------------
%  rectpulse_local — repete cada símbolo N vezes (pulso retangular)
%    Equivalente a: rectpulse(x, N)
% ---------------------------------------------------------
function y = rectpulse_local(x, N)
    y = repelem(x, N);
end

% ---------------------------------------------------------
%  intdump_local — integra e despeja a cada N amostras
%    Equivalente a: intdump(x, N)
% ---------------------------------------------------------
function y = intdump_local(x, N)
    L = floor(length(x) / N);
    x = x(1 : L * N);
    y = sum(reshape(x, N, L), 1);
end

% ---------------------------------------------------------
%  eyediagram_local — diagrama de olho sem toolbox
%    eyediagram_local(x, spp)
%      x   : sinal (vetor)
%      spp : amostras por período do símbolo (janela = spp)
%
%  Sobrepõe cada janela de 'spp' amostras, gerando o padrão
%  de olho. A linha vermelha tracejada marca o limiar de decisão.
% ---------------------------------------------------------
function eyediagram_local(x, spp)
    x   = x(:)';
    N   = length(x);
    n_j = floor(N / spp);          % número de janelas completas

    x = x(1 : n_j * spp);
    M = reshape(x, spp, n_j);      % spp linhas × n_j colunas

    t_eye = 0 : spp-1;

    hold on;
    for col = 1 : n_j
        % RGBA: cor azul com alfa 0.15 para sobreposição suave
        % Octave aceita vetor [R G B] (sem alfa) — a transparência
        % será ignorada graciosamente, mantendo a cor
        plot(t_eye, M(:, col)', 'Color', [0.2 0.4 0.8], ...
             'LineWidth', 0.5);
    end
    % limiar de decisão
    plot([0 spp-1], [0 0], 'r--', 'LineWidth', 1.5);
    hold off;

    grid on;
    xlim([0 spp-1]);
end
