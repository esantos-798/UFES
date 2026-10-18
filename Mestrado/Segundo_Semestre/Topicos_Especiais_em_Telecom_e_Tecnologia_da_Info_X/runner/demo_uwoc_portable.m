% demo_uwoc_portable.m  -- downlink 5G NR sobre o canal UWOC, MATLAB ou Octave
% Usa o 5G Toolbox se existir; caso contrario, as funcoes custom_* (nenhum toolbox necessario).
% Todos os parametros ficam aqui; a logica esta em run_downlink.m.

%% Parametros
pBackend = 'auto';              % 'auto' | 'toolbox' | 'custom'
pChannel = 'uwoc-b2b';          % 'software' | 'b2b' | 'uwoc-b2b'
pDigitalModulation = '16QAM';   % 'QPSK' | '16QAM' | '64QAM'
pSCS = 15;                      % kHz
pSizeGrid = 25;                 % RBs
pSNRdB = 50;                    % so no canal 'software' (ou com pUWOC.SNR_ref = 'software')
pVLCCenterFrequency = 5e6;      % portadora do DUC (Hz); mantenha fc >= K*SCS/2 e fc + K*SCS/2 < pDACSampleRate/2
pDACSampleRate = 75e6;          % taxa do DAC (Hz)
pSeed = 42;

% UWOC (campos omitidos usam uwoc_defaults.m)
pUWOC = struct('SNR_dB', 30, ...            % SNR (dB), Inf = sem ruido
               'SNR_ref', 'rx', ...         % 'electrical' | 'rx' (subportadoras uteis) | 'software' (usa pSNRdB)
               'pdLaw', 'linear', ...       % 'linear' | 'square' (replica SimLED_conv.m)
               'I_mod_A', 50);              % pico do sinal modulante (mA); excursao = alpha*I_mod_A

%% Execucao
cfg = struct('backend', pBackend, 'pChannel', pChannel, 'pDigitalModulation', pDigitalModulation, ...
             'pSCS', pSCS, 'pSizeGrid', pSizeGrid, 'pSNRdB', pSNRdB, ...
             'pVLCCenterFrequency', pVLCCenterFrequency, 'pDACSampleRate', pDACSampleRate, ...
             'uwoc', pUWOC, 'seed', pSeed);
R = run_downlink(cfg);
print_downlink_result(R);
