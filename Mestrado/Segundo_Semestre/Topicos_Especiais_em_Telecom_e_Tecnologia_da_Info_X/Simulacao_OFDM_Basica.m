% =========================================================================
% SIMULAÇÃO OFDM 5G NR - BÁSICA (Híbrida/Universal)
% 
% Compatível com:
% 1. MATLAB com 5G Toolbox (Baseado no projeto '5g-ofdm-test' do Augusto)
% 2. MATLAB Versão Free (Sem toolboxes proprietárias)
% 3. GNU Octave
% =========================================================================
clear; close all; clc;

% =========================================================================
% 1. DETECÇÃO DE AMBIENTE
% =========================================================================
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
has5GToolbox = exist('nrOFDMModulate', 'file') > 0;

disp('======================================================');
disp('   SIMULAÇÃO OFDM 5G NR - BÁSICA (TX -> AWGN -> RX)   ');
disp('======================================================');
if isOctave
    disp('[Ambiente]: GNU Octave');
    % Suprime mensagens de erro caso os pacotes não existam
    try pkg load signal communications; catch; end; 
else
    disp('[Ambiente]: MATLAB');
end

if has5GToolbox
    disp('[Modo]: 5G Toolbox nativo detectado (Metodologia Augusto).');
else
    disp('[Modo]: Emulação manual equivalente (5G Toolbox ausente).');
end

% =========================================================================
% 2. PARÂMETROS COMUNS (Numerologia 5G: SCS = 15 kHz, 5 MHz BW)
% =========================================================================
SCS = 15e3;                % Espaçamento de subportadora (15 kHz)
N_RBs = 25;                % Número de Resource Blocks (25 * 12 = 300 subp.)
SNR_dB = 18;               % Relação Sinal-Ruído (SNR) no canal AWGN
M = 16;                    % Modulação (16-QAM)
N_symbols = 14;            % Símbolos por slot (1 slot temporal)

% =========================================================================
% 3. TRANSMISSOR (TX)
% =========================================================================
if has5GToolbox
    % --- ROTA 1: COM 5G TOOLBOX (Exato padrão da dissertação) ---
    carrier = nrCarrierConfig('SubcarrierSpacing', SCS/1000, 'NSizeGrid', N_RBs);
    info = nrOFDMInfo(carrier);
    Fs = info.SampleRate;
    
    % Configura um canal PDSCH ocupando todo o grid (Simplificado)
    pdsch = nrPDSCHConfig('PRBSet', 0:(N_RBs-1), 'Modulation', '16QAM');
    [ind, pdschInfo] = nrPDSCHIndices(carrier, pdsch);
    
    % Gera os símbolos e mapeia na grade
    bits = randi([0 1], pdschInfo.G, 1);
    tx_syms = nrPDSCH(carrier, pdsch, bits);
    
    tx_grid = nrResourceGrid(carrier);
    tx_grid(ind) = tx_syms;
    
    % Modula para o domínio do tempo com IFFT nativa do 5G
    [tx_waveform, ~] = nrOFDMModulate(carrier, tx_grid);
    
else
    % --- ROTA 2: SEM 5G TOOLBOX (MATLAB Free / Octave) ---
    N_fft = 512;
    Fs = SCS * N_fft;      % Taxa de amostragem de 7.68 MHz
    CP_len = 36;           % Prefixo Cíclico
    N_active = N_RBs * 12; % 300 subportadoras ativas
    
    % Gera símbolos 16-QAM manualmente para não depender do 'Communications Toolbox'
    data = randi([0 15], N_active * N_symbols, 1);
    % Dicionário 16-QAM normalizado (Potência média = 1)
    map16 = [-3-3i, -3-1i, -3+3i, -3+1i, -1-3i, -1-1i, -1+3i, -1+1i, ...
              3-3i,  3-1i,  3+3i,  3+1i,  1-3i,  1-1i,  1+3i,  1+1i] / sqrt(10);
    tx_syms = map16(data + 1).';
    
    % Mapeia na grade OFDM (Deixando DC e guard bands em zero)
    tx_grid = zeros(N_fft, N_symbols);
    idx_active = [2:(N_active/2 + 1), (N_fft - N_active/2 + 1):N_fft];
    tx_grid(idx_active, :) = reshape(tx_syms, N_active, N_symbols);
    
    % IFFT e adição de Prefixo Cíclico manual
    time_syms = ifft(tx_grid, N_fft) * sqrt(N_fft);
    time_syms_cp = [time_syms(end-CP_len+1:end, :); time_syms];
    tx_waveform = time_syms_cp(:);
end

% Normaliza o sinal transmitido
tx_waveform = tx_waveform / max(abs(tx_waveform));

% =========================================================================
% 4. CANAL (AWGN)
% =========================================================================
% Calculando e inserindo ruído manualmente (Substitui a função awgn())
% Isso garante que o código rode em qualquer versão "Free"
sigPower = mean(abs(tx_waveform).^2);
noisePower = sigPower / (10^(SNR_dB/10));
noise = sqrt(noisePower/2) * (randn(size(tx_waveform)) + 1i*randn(size(tx_waveform)));

rx_waveform = tx_waveform + noise;

% =========================================================================
% 5. RECEPTOR (RX)
% =========================================================================
if has5GToolbox
    % Demodula o OFDM usando a função nativa do 5G NR
    rx_grid = nrOFDMDemodulate(carrier, rx_waveform);
    rx_syms = rx_grid(ind); % Extrai apenas as subportadoras úteis
else
    % Demodula OFDM manualmente
    rx_sync = rx_waveform(1: (N_fft+CP_len)*N_symbols); % Ajuste
    rx_syms_cp = reshape(rx_sync, N_fft + CP_len, N_symbols);
    rx_syms_no_cp = rx_syms_cp(CP_len+1:end, :);
    
    % FFT para voltar ao domínio da frequência
    rx_grid = fft(rx_syms_no_cp, N_fft) / sqrt(N_fft);
    rx_syms = reshape(rx_grid(idx_active, :), [], 1);
    
    % Equalização simples baseada em média de fase (Flat Fading)
    H_mean = mean(rx_syms ./ tx_syms);
    rx_syms = rx_syms / H_mean;
end

% Cálculo de Métricas (EVM e Estimativa de SNR Prática)
error_vector = rx_syms - tx_syms;
evm_rms = sqrt(mean(abs(error_vector).^2) / mean(abs(tx_syms).^2)) * 100;

SNR_estimada = 10*log10(mean(abs(tx_syms).^2) / mean(abs(error_vector).^2));

fprintf('\n--- RESULTADOS ---\n');
fprintf('SNR Aplicada no Canal AWGN : %5.2f dB\n', SNR_dB);
fprintf('SNR Recuperada nos Símbolos: %5.2f dB\n', SNR_estimada);
fprintf('EVM RMS da Constelação     : %5.2f %%\n', evm_rms);

% =========================================================================
% 6. VISUALIZAÇÃO DOS SINAIS E ESPECTROS
% =========================================================================
figure('Name', 'Analise OFDM 5G NR (Basica)', 'Position', [100, 100, 1200, 400]);

% Espectro TX e RX criados usando a FFT básica (Funciona em todos os ambientes)
N_plot = length(tx_waveform);
f_axis = linspace(-Fs/2, Fs/2, N_plot)/1e6;

Tx_spec = 20*log10(abs(fftshift(fft(tx_waveform))));
Tx_spec = Tx_spec - max(Tx_spec); % Normaliza pico para 0 dB

Rx_spec = 20*log10(abs(fftshift(fft(rx_waveform))));
Rx_spec = Rx_spec - max(Tx_spec); % Mantém relatividade ao pico Tx

% [Plot 1] Espectro TX
subplot(1, 3, 1);
plot(f_axis, Tx_spec, 'b', 'LineWidth', 1.2);
title('Espectro Tx (Limpo)');
xlabel('Frequência (MHz)'); ylabel('Magnitude (dB)');
axis([-Fs/2e6 Fs/2e6 -60 5]); grid on;

% [Plot 2] Espectro RX com AWGN
subplot(1, 3, 2);
plot(f_axis, Rx_spec, 'r', 'LineWidth', 1.2);
title(['Espectro Rx (SNR = ', num2str(SNR_dB), ' dB)']);
xlabel('Frequência (MHz)'); ylabel('Magnitude (dB)');
axis([-Fs/2e6 Fs/2e6 -60 5]); grid on;

% [Plot 3] Constelação RX
subplot(1, 3, 3);
plot(real(rx_syms), imag(rx_syms), 'o', 'Color', [0.8500 0.3250 0.0980], 'MarkerSize', 4); hold on;
plot(real(tx_syms), imag(tx_syms), 'kx', 'MarkerSize', 8, 'LineWidth', 2);
title(['Constelação RX (EVM: ', num2str(evm_rms, '%.2f'), '%)']);
legend('Símbolos Recebidos', 'Referência (Tx)', 'Location', 'best');
axis square; grid on;

disp('Execução do gráfico concluída!');
