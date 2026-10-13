% =========================================================================
% Simulação 5G NR OFDM sobre canal UWOC (Underwater Optical Wireless Comm)
% Integração da tese (ADALM), modelo físico de LED e canal Dupla Gama.
% =========================================================================
clear; close all; clc;

% Adicionando caminhos para as bibliotecas do projeto base
addpath('5g-ofdm-test/src/adalm2000');
addpath('5g-ofdm-test/src/adalm-pluto');
addpath('5g-ofdm-test/src/etc');
addpath('5g-ofdm-test/src/filters');
addpath('5g-ofdm-test/src/newradio');

%% 1. Parâmetros do Sistema

% 1.1. Parâmetros 5G NR
pSCS = 15;                      % Espaçamento de subportadora (kHz)
pSizeGrid = 25;                 % Número de Resource Blocks (RBs)
pLayers = 1;                    % Número de camadas espaciais
pDigitalModulation = '16QAM';   % Modulação digital ('QPSK', '16QAM', '64QAM')
pNCellID = 102;                 % Cell ID para sincronização
pCodeRate = 490/1024;           % Taxa de código

% 1.2. Parâmetros do Conversor e DUC
pDACSampleRate = 75e6;          % Taxa de amostragem do DAC (75 MSps)
pVLCCenterFrequency = 5e6;      % Frequência da Portadora Digital (5 MHz)

% 1.3. Parâmetros do LED Não-Linear
Lum_max = 755;                  % Iluminância máxima [lux] ou pot. óptica rel.
zeta  = 20;                     % Parâmetro de saturação (adimensional)
k     = 1.9;                    % Fator de forma
I_bias = 800;                   % Corrente de polarização DC (Bias) [mA]
alpha = 250;                    % Ganho de amplitude do sinal OFDM 

% 1.4. Parâmetros do Canal UWOC (Águas Costeiras, AL = 8) e Ruído
C1 = 0.3958; C2 = 8.811; C3 = 1589; C4 = 51.35;
SNR_dB = 30;                    % Relação Sinal-Ruído (EbNo/SNR) no receptor

%% 2. Geração da Forma de Onda 5G NR (Banda Base)
disp('Gerando grid 5G NR e forma de onda OFDM...');
cCarrier = nrCarrierConfig('SubcarrierSpacing', pSCS, 'NSizeGrid', pSizeGrid);
cInfo = nrOFDMInfo(cCarrier);
cSampleRate = cInfo.SampleRate;

% Construindo Canais (SSB, PDSCH, PDCCH) baseados na tese
ssbBlockGrid = initSSB(pNCellID, cCarrier, false);
pdsch = nrPDSCHConfig('NumLayers', pLayers, 'PRBSet', 0:(pSizeGrid-1), 'Modulation', pDigitalModulation);
[pdschIndices, pdschInfo] = nrPDSCHIndices(cCarrier, pdsch);
[encodeDLSCH, trBlkSizes] = initDLSCH(pCodeRate, pdsch, cCarrier);
trBlk = randi([0 1], trBlkSizes, 1);
setTransportBlock(encodeDLSCH, trBlk);
codedTrBlock = encodeDLSCH(pdsch.Modulation, 1, pdschInfo.G, 0);
pdschSymbols = nrPDSCH(cCarrier, pdsch, codedTrBlock);
pdschGrid = mapPDSCH(cCarrier, pdsch, pdschSymbols, pdschIndices, pLayers, false);

pdcch = nrPDCCHConfig('NStartBWP', 0, 'NSizeBWP', pSizeGrid);
pdcch.AggregationLevel = 4; pdcch.CORESET.Duration = 1; pdcch.CORESET.FrequencyResources = [1,1,1,1];
dciBits = randi([0 1], 64, 1, 'int8');
pdcchGrid = mapPDCCH(cCarrier, pdcch, dciBits, false);

DLgrid = mapDLGrid(ssbBlockGrid, pdschGrid, pdcchGrid, cCarrier, false);

% Modulação OFDM (Sinal Complexo em Banda Base)
[waveform_bb, cInfo] = nrOFDMModulate(cCarrier, DLgrid, 'Windowing', 0);
waveform_bb = waveform_bb / max(abs(waveform_bb)); % Normalização

%% 3. Transmissor Óptico: Upsampling e DUC (Passa-Faixa)
disp('Realizando DUC (Digital Up-Conversion) para banda passante...');
% Upsampling para a taxa do DAC
[P1bb, Q1bb] = rat(pDACSampleRate / cSampleRate);
waveform_up = resample(waveform_bb, P1bb, Q1bb);

% Modulação QAM / Mistura com portadora (Sinal Real)
tx_passband = modulate(real(waveform_up), pVLCCenterFrequency, pDACSampleRate, 'qam', imag(waveform_up));
tx_passband = tx_passband / max(abs(tx_passband));

%% 4. Modelagem Física do LED
disp('Aplicando distorção não-linear do LED...');
% Adiciona Corrente de Bias e amplifica o sinal
I_sig = I_bias + (alpha * tx_passband);

% Evita correntes negativas extremas no LED (Clipping inferior hard)
I_sig(I_sig < 0) = 0;

% Passa pela Curva do LED
tx_optical = I_sig ./ (zeta + (I_sig / Lum_max).^(2*k)).^(1/(2*k));

% Remove o ganho DC estático apenas para visualização da potência variada na transmissão
tx_optical_ac = tx_optical - mean(tx_optical); 

%% 5. Canal UWOC: Resposta ao Impulso (Double Gamma)
disp('Aplicando canal UWOC (Double Gamma) e ruído...');

% Gerando a resposta ao impulso em alta resolução (10 GHz)
Fs_high = 10e9; 
t_ns_high = 0:0.1:16; % 16 ns de canal
h_uwoc_high = C1 .* t_ns_high .* exp(-C2 .* t_ns_high) + C3 .* t_ns_high .* exp(-C4 .* t_ns_high);
h_uwoc_high = h_uwoc_high / sum(h_uwoc_high);

% Resampling do canal para a taxa do DAC (75 MSps) para evitar gargalo de RAM
[P_ch, Q_ch] = rat(pDACSampleRate / Fs_high);
h_uwoc_75m = resample(h_uwoc_high, P_ch, Q_ch);
h_uwoc_75m = h_uwoc_75m / sum(h_uwoc_75m); % Renormaliza ganho unitário

% Convolução do sinal óptico com o canal
rx_optical = conv(tx_optical_ac, h_uwoc_75m, 'same');

% Ruído Térmico no Receptor (AWGN)
rx_electrical = awgn(rx_optical, SNR_dB, 'measured');

%% 6. Receptor Óptico: DDC (Banda Base) e Downsampling
disp('Demodulando sinal no receptor...');

% Normaliza e remove resíduos DC
rx_electrical = rx_electrical / max(abs(rx_electrical));
rx_electrical = rx_electrical - mean(rx_electrical);

% DDC: Demodulação QAM passa-faixa para banda base
[rxReal, rxImag] = demod(rx_electrical, pVLCCenterFrequency, pDACSampleRate, 'qam');
rx_bb_up = rxReal + 1i.*rxImag;

% Downsampling de volta para a taxa do 5G NR
rx_bb = resample(rx_bb_up, Q1bb, P1bb);

%% 7. Demodulação 5G NR e Métricas
disp('Decodificando grid 5G e extraindo EVM...');

% Demodula OFDM
rxGrid = nrOFDMDemodulate(cCarrier, rx_bb);

% Extraindo Símbolos PDSCH Recebidos usando os mesmos índices
rx_pdschSymbols = rxGrid(pdschIndices);

% Medição do EVM
evm = comm.EVM;
evm_rms = evm(pdschSymbols, rx_pdschSymbols);
fprintf('\n--- RESULTADOS ---\n');
fprintf('EVM (RMS) no PDSCH: %.2f %%\n', evm_rms);

%% 8. Gráficos de Avaliação
figure('Name', 'Avaliação do Enlace 5G UWOC', 'Position', [100, 100, 1200, 800]);

% Gráfico 1: Curva Não-Linear do LED
subplot(2, 2, 1);
I_sweep = linspace(0, 2000, 500);
Lum_sweep = I_sweep ./ (zeta + (I_sweep / Lum_max).^(2*k)).^(1/(2*k));
plot(I_sweep, Lum_sweep, 'b', 'LineWidth', 2); hold on;
plot(I_bias, I_bias ./ (zeta + (I_bias / Lum_max).^(2*k)).^(1/(2*k)), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
title('Curva do LED e Ponto de Polarização (Bias)');
xlabel('Corrente de Entrada (mA)'); ylabel('Potência Óptica');
grid on;

% Gráfico 2: Resposta ao Impulso do Canal UWOC
subplot(2, 2, 2);
plot(t_ns_high, h_uwoc_high, 'LineWidth', 2);
title('Canal UWOC (Double Gamma) - AL=8');
xlabel('Tempo (ns)'); ylabel('Amplitude');
grid on;

% Gráfico 3: Espectro do Sinal Transmitido (Passa-Faixa)
subplot(2, 2, 3);
pwelch(tx_passband, [], [], [], pDACSampleRate, 'centered');
title('Densidade Espectral do Sinal DUC (Passa-Faixa)');

% Gráfico 4: Constelação PDSCH
subplot(2, 2, 4);
plot(rx_pdschSymbols, 'o', 'Color', [0 0.4470 0.7410], 'MarkerSize', 4); hold on;
plot(pdschSymbols, 'rx', 'MarkerSize', 6);
title(['Constelação PDSCH (EVM: ', num2str(evm_rms, '%.2f'), '%)']);
legend('Símbolos Rx', 'Símbolos Tx (Referência)', 'Location', 'best');
axis square; grid on;

disp('Simulação concluída com sucesso!');
