% =========================================================================
% Simulação OFDM (Emulando Numerologia 5G NR) sobre canal UWOC para Octave
% =========================================================================
clear; close all; clc;

% Carregar pacotes necessários no Octave
try
    pkg load signal;
    pkg load communications;
catch
    disp('Aviso: Certifique-se de ter os pacotes "signal" e "communications" instalados no Octave.');
end

disp('--- Iniciando Simulação (GNU Octave) ---');

%% 1. Parâmetros do Sistema

% 1.1. Emulação da Numerologia 5G NR (SCS 15 kHz, 5 MHz BW)
SCS = 15e3;                     % Espaçamento de subportadora (15 kHz)
N_RBs = 25;                     % Número de Resource Blocks (25 RBs)
N_active_subcarriers = N_RBs * 12; % 300 subportadoras ativas
N_fft = 512;                    % Tamanho da FFT
Fs_bb = SCS * N_fft;            % Taxa de amostragem Banda Base (7.68 MSps)
CP_len = 36;                    % Tamanho do Prefixo Cíclico
N_symbols = 14;                 % Número de símbolos OFDM (1 slot 5G)
M_QAM = 16;                     % Modulação (16-QAM)

% 1.2. Parâmetros do Conversor e DUC
Fs_dac = 75e6;                  % Taxa de amostragem do DAC (75 MSps)
Fc = 5e6;                       % Frequência da Portadora Digital (5 MHz)

% 1.3. Parâmetros do LED Não-Linear
Lum_max = 755; zeta = 20; k = 1.9;
I_bias = 800; alpha = 250;

% 1.4. Parâmetros do Canal UWOC (Águas Costeiras, AL = 8)
C1 = 0.3958; C2 = 8.811; C3 = 1589; C4 = 51.35;
SNR_dB = 30;

%% 2. Geração da Forma de Onda OFDM (Banda Base)
disp('Gerando grid OFDM e símbolos no domínio do tempo...');
% Gera dados aleatórios e modula em QAM
data = randi([0, M_QAM-1], N_active_subcarriers * N_symbols, 1);
qam_syms = qammod(data, M_QAM);

% Mapeia os símbolos nas subportadoras ativas da FFT (pula a portadora DC)
tx_grid = zeros(N_fft, N_symbols);
% Índices simétricos ao redor da DC (índice 1)
idx_active = [2:(N_active_subcarriers/2 + 1), (N_fft - N_active_subcarriers/2 + 1):N_fft];
tx_grid(idx_active, :) = reshape(qam_syms, N_active_subcarriers, N_symbols);

% IFFT e Prefixo Cíclico
time_syms = ifft(tx_grid, N_fft) * sqrt(N_fft);
time_syms_cp = [time_syms(end-CP_len+1:end, :); time_syms]; % Adiciona CP
tx_bb = time_syms_cp(:);
tx_bb = tx_bb / max(abs(tx_bb)); % Normalização

%% 3. Transmissor Óptico: Upsampling e DUC (Passa-Faixa)
disp('Realizando DUC (Digital Up-Conversion) para banda passante...');
[P1bb, Q1bb] = rat(Fs_dac / Fs_bb);
tx_up = resample(tx_bb, P1bb, Q1bb);

% Modulação Passa-Faixa Matemática (Mestura com portadora complexa)
t_up = (0:length(tx_up)-1)' / Fs_dac;
tx_passband = real(tx_up .* exp(1i * 2 * pi * Fc * t_up));
tx_passband = tx_passband / max(abs(tx_passband));

%% 4. Modelagem Física do LED
disp('Aplicando distorção não-linear do LED...');
I_sig = I_bias + (alpha * tx_passband);
I_sig(I_sig < 0) = 0; % Clipping
tx_optical = I_sig ./ (zeta + (I_sig / Lum_max).^(2*k)).^(1/(2*k));
tx_optical_ac = tx_optical - mean(tx_optical); 

%% 5. Canal UWOC: Resposta ao Impulso (Double Gamma)
disp('Aplicando canal UWOC (Double Gamma) e ruído...');
Fs_high = 10e9; 
t_ns_high = 0:0.1:16; 
h_uwoc_high = C1 .* t_ns_high .* exp(-C2 .* t_ns_high) + C3 .* t_ns_high .* exp(-C4 .* t_ns_high);
h_uwoc_high = h_uwoc_high / sum(h_uwoc_high);

[P_ch, Q_ch] = rat(Fs_dac / Fs_high);
h_uwoc_75m = resample(h_uwoc_high, P_ch, Q_ch);
h_uwoc_75m = h_uwoc_75m / sum(h_uwoc_75m);

rx_optical = conv(tx_optical_ac, h_uwoc_75m, 'same');

if exist('awgn', 'file')
    rx_electrical = awgn(rx_optical, SNR_dB, 'measured');
else
    % Fallback se 'awgn' não estiver disponível
    P_sig = var(rx_optical);
    rx_electrical = rx_optical + sqrt(P_sig/(10^(SNR_dB/10))) * randn(size(rx_optical));
end

%% 6. Receptor Óptico: DDC (Banda Base) e Downsampling
disp('Demodulando sinal no receptor...');
rx_electrical = rx_electrical / max(abs(rx_electrical));
rx_electrical = rx_electrical - mean(rx_electrical);

% DDC: Mistura com exponencial complexa negativa
rx_mix = rx_electrical .* exp(-1i * 2 * pi * Fc * t_up);

% Filtro Passa-Baixa (Recuperar apenas banda base)
[b, a] = butter(4, (Fc * 0.8) / (Fs_dac / 2));
rx_bb_up = 2 * filter(b, a, rx_mix); % Ganho 2x pois metade da energia foi filtrada

% Downsampling de volta
rx_bb_demod = resample(rx_bb_up, Q1bb, P1bb);

%% 7. Sincronização e Demodulação OFDM
disp('Decodificando OFDM e extraindo EVM...');
% Sincronização temporal simples (Cross-Correlation)
[corr_val, lags] = xcorr(rx_bb_demod, tx_bb);
[~, max_idx] = max(abs(corr_val));
delay = lags(max_idx);

if delay > 0
    rx_sync = rx_bb_demod(delay+1:end);
elseif delay < 0
    rx_sync = [zeros(-delay, 1); rx_bb_demod];
else
    rx_sync = rx_bb_demod;
end
rx_sync = rx_sync(1:length(tx_bb));

% Remove CP e FFT
rx_syms_cp = reshape(rx_sync, N_fft + CP_len, N_symbols);
rx_syms = rx_syms_cp(CP_len+1:end, :);
rx_grid = fft(rx_syms, N_fft) / sqrt(N_fft);
rx_qam_syms = reshape(rx_grid(idx_active, :), [], 1);

% Equalização Zero-Forcing (Média do ganho e fase do canal)
% O filtro DDC e o Upsampling causam defasagem fixa, mitigamos isso aqui
H_mean = mean(rx_qam_syms ./ qam_syms);
rx_qam_syms_eq = rx_qam_syms / H_mean;

% Cálculo do EVM (RMS)
error_vector = rx_qam_syms_eq - qam_syms;
evm_rms = sqrt(mean(abs(error_vector).^2) / mean(abs(qam_syms).^2)) * 100;
fprintf('\n--- RESULTADOS ---\n');
fprintf('EVM (RMS): %.2f %%\n', evm_rms);

%% 8. Gráficos de Avaliação
figure('Name', 'Avaliação do Enlace OFDM UWOC (Octave)', 'Position', [100, 100, 1200, 800]);

% Gráfico 1: Curva do LED
subplot(2, 2, 1);
I_sweep = linspace(0, 2000, 500);
Lum_sweep = I_sweep ./ (zeta + (I_sweep / Lum_max).^(2*k)).^(1/(2*k));
plot(I_sweep, Lum_sweep, 'b', 'LineWidth', 2); hold on;
plot(I_bias, I_bias ./ (zeta + (I_bias / Lum_max).^(2*k)).^(1/(2*k)), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
title('Curva do LED e Ponto de Bias');
xlabel('Corrente (mA)'); ylabel('Potência Óptica');
grid on;

% Gráfico 2: Resposta ao Impulso UWOC
subplot(2, 2, 2);
plot(t_ns_high, h_uwoc_high, 'LineWidth', 2);
title('Canal UWOC (Double Gamma) - AL=8');
xlabel('Tempo (ns)'); ylabel('Amplitude');
grid on;

% Gráfico 3: Espectro Passa-Faixa
subplot(2, 2, 3);
N_spec = length(tx_passband);
f_axis = linspace(-Fs_dac/2, Fs_dac/2, N_spec);
X_pass = fftshift(fft(tx_passband));
plot(f_axis/1e6, 20*log10(abs(X_pass)/max(abs(X_pass))));
title('Espectro DUC (Passa-Faixa)');
xlabel('Frequência (MHz)'); ylabel('Magnitude (dB)');
axis([0 15 -60 5]); grid on;

% Gráfico 4: Constelação
subplot(2, 2, 4);
plot(real(rx_qam_syms_eq), imag(rx_qam_syms_eq), 'ob', 'MarkerSize', 4); hold on;
plot(real(qam_syms), imag(qam_syms), 'xr', 'MarkerSize', 6);
title(['Constelação (EVM: ', num2str(evm_rms, '%.2f'), '%)']);
legend('Rx', 'Tx', 'Location', 'best');
axis square; grid on;

disp('Simulação concluída!');
