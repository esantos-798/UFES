% demo_uwoc_matlab.m  -- downlink 5G NR sobre o canal UWOC, MATLAB COM 5G Toolbox
% Exige o 5G Toolbox e as funcoes do laboratorio (initSSB, mapDLGrid, mapPDSCH, initDLSCH, ...):
% o BER e pos-LDPC. Para rodar sem toolbox (ou em Octave), use demo_uwoc_portable.m.
% Todos os parametros ficam aqui; a logica esta em run_downlink.m.

%% Parametros
pChannel = 'uwoc-b2b';          % 'software' | 'b2b' | 'uwoc-b2b'
pDigitalModulation = '16QAM';   % 'QPSK' | '16QAM' | '64QAM'
pSCS = 15;                      % kHz
pSizeGrid = 25;                 % RBs
pSNRdB = 50;                    % so no canal 'software' (ou com pUWOC.SNR_ref = 'software')
pVLCCenterFrequency = 5e6;      % portadora do DUC (Hz)
pDACSampleRate = 75e6;          % taxa do DAC (Hz)
pSeed = 42;

pUWOC = struct('SNR_dB', 30, ...
               'SNR_ref', 'rx', ...
               'pdLaw', 'linear', ...
               'I_mod_A', 50);

%% Execucao
cfg = struct('backend', 'toolbox', 'pChannel', pChannel, 'pDigitalModulation', pDigitalModulation, ...
             'pSCS', pSCS, 'pSizeGrid', pSizeGrid, 'pSNRdB', pSNRdB, ...
             'pVLCCenterFrequency', pVLCCenterFrequency, 'pDACSampleRate', pDACSampleRate, ...
             'uwoc', pUWOC, 'seed', pSeed);
R = run_downlink(cfg);
print_downlink_result(R);
