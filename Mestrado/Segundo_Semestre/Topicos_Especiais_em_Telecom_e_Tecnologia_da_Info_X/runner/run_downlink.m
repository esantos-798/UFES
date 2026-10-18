function R = run_downlink(cfg)
%RUN_DOWNLINK Uma execucao do downlink 5G NR: TX + canal + RX + metricas, com backend selecionavel.
%
%   R = run_downlink(cfg)
%
%   Funciona em MATLAB e Octave. Nao e um script: devolve uma struct de metricas, entao pode ser
%   chamada em lacos (varreduras de SNR, modulacao, numerologia) e testada.
%
%   cfg (todos os campos sao opcionais):
%     backend             'auto' (padrao) | 'toolbox' | 'custom'
%                           auto    : 5G Toolbox se existir (nrOFDMModulate), senao funcoes custom_*
%                           toolbox : exige 5G Toolbox + funcoes do laboratorio (initSSB, mapDLGrid, ...)
%                           custom  : sempre as funcoes custom_* (nenhum toolbox; Octave com pacote signal)
%     pChannel            'software' | 'b2b' | 'uwoc-b2b'
%     pDigitalModulation  'QPSK' | '16QAM' | '64QAM'  (o nivel M e derivado daqui)
%     pSCS (15) [kHz], pSizeGrid (25), pNCellID (102), pCodeRate (490/1024), pLayers (1)
%     pSNRdB (50), pFrequencyOffset (1.2e3), pPhaseOffset (15), pMaxRandomOffset (100)   % canal 'software'
%     pVLCCenterFrequency (5e6), pDACSampleRate (75e6)                                   % b2b e uwoc-b2b
%     uwoc                struct parcial de uwoc_defaults (SNR_dB, SNR_ref, pdLaw, I_mod_A, ...)
%                         K, SCS_Hz, fc e Fs_dac saem de pSizeGrid, pSCS, pVLCCenterFrequency e pDACSampleRate;
%                         com SNR_ref = 'software', SNR_dB passa a ser pSNRdB
%     seed (42), verbose (false)
%
%   R (principais): ber, berType, evm_pct, evm_noDC_pct, evm_noDCnb_pct, snr_rx_dB, timingOffset,
%                   cfo_coarse_hz, cfo_fine_hz, backend, channelNotes, uwoc (diagnosticos), ...
%   evm_pct usa o evm do laboratorio; as variantes excluem a subportadora de DC e a faixa DC +- 2
%   (onde a remocao de media do script corrompe dados e o piloto de DMRS).
%   snr_rx_dB: SNR nas subportadoras uteis (comparavel entre canais; no 'software' e medida do ruido somado).

    if nargin < 1, cfg = struct(); end
    cfg = fill_defaults(cfg);

    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    if isOctave
        try, pkg load signal communications; catch, end
    end
    addLabPaths();

    % ---------------------------------------------------------------- backend
    hasToolbox = exist('nrOFDMModulate', 'file') > 0;
    switch lower(cfg.backend)
        case 'auto',    useToolbox = hasToolbox;
        case 'custom',  useToolbox = false;
        case 'toolbox'
            if ~hasToolbox
                error('run_downlink:NoToolbox', 'backend ''toolbox'' pedido, mas o 5G Toolbox nao foi encontrado.');
            end
            useToolbox = true;
        otherwise
            error('run_downlink:Backend', 'backend deve ser ''auto'', ''toolbox'' ou ''custom''.');
    end
    if useToolbox
        needed = {'initSSB', 'mapPDSCH', 'mapPDCCH', 'mapDLGrid', 'initDLSCH', 'nrOFDM_CFO', 'detectPSS', 'detecSSS', 'evm'};
        missing = needed(cellfun(@(n) exist(n, 'file') == 0, needed));
        if ~isempty(missing)
            error('run_downlink:LabFunctions', 'funcoes do laboratorio ausentes no path: %s', strjoin(missing, ', '));
        end
    end
    hasModDemod = exist('modulate', 'file') > 0 && exist('demod', 'file') > 0;

    % ---------------------------------------------------------------- parametros derivados
    modNames  = {'QPSK', '16QAM', '64QAM'};  modLevels = [4 16 64];
    mi = find(strcmpi(cfg.pDigitalModulation, modNames), 1);
    if isempty(mi), error('run_downlink:Mod', 'Modulacao nao suportada: %s', cfg.pDigitalModulation); end
    M = modLevels(mi);
    if isfield(cfg, 'pModulationLevel') && cfg.pModulationLevel ~= M
        error('run_downlink:ModLevel', 'pModulationLevel (%d) nao corresponde a %s (M = %d).', cfg.pModulationLevel, cfg.pDigitalModulation, M);
    end
    pSCS = cfg.pSCS;  pSizeGrid = cfg.pSizeGrid;  pNCellID = cfg.pNCellID;  K = 12 * pSizeGrid;

    seedRandom(cfg.seed);

    % ---------------------------------------------------------------- portadora e TX
    if useToolbox
        cCarrier = nrCarrierConfig('SubcarrierSpacing', pSCS, 'NSizeGrid', pSizeGrid);
        cInfo = nrOFDMInfo(cCarrier);
        ssbBlockGrid = initSSB(pNCellID, cCarrier, false);
        pdsch = nrPDSCHConfig('NumLayers', cfg.pLayers, 'PRBSet', 0:(pSizeGrid-1), 'Modulation', cfg.pDigitalModulation);
        [pdschIndices, pdschInfo] = nrPDSCHIndices(cCarrier, pdsch);
        [encodeDLSCH, trBlkSizes] = initDLSCH(cfg.pCodeRate, pdsch, cCarrier);
        trBlk = randi([0 1], trBlkSizes, 1);
        setTransportBlock(encodeDLSCH, trBlk);
        codedTrBlock = encodeDLSCH(pdsch.Modulation, 1, pdschInfo.G, 0);
        pdschSymbols = nrPDSCH(cCarrier, pdsch, codedTrBlock);
        pdschGrid = mapPDSCH(cCarrier, pdsch, pdschSymbols, pdschIndices, cfg.pLayers, false);
        pdcch = nrPDCCHConfig('NStartBWP', 0, 'NSizeBWP', pSizeGrid);
        pdcch.AggregationLevel = 4;
        pdcch.CORESET.Duration = 1;
        pdcch.CORESET.FrequencyResources = [1, 1, 1, 1];
        dciBits = randi([0 1], 64, 1, 'int8');
        pdcchGrid = mapPDCCH(cCarrier, pdcch, dciBits, false);
        DLgrid = mapDLGrid(ssbBlockGrid, pdschGrid, pdcchGrid, cCarrier, false);
        [waveform, cInfo] = nrOFDMModulate(cCarrier, DLgrid, 'Windowing', 0);
        txInfo = [];
        txSymbols = pdschSymbols;
        berType = 'pos-LDPC (5G Toolbox)';
    else
        cCarrier = custom_nrCarrierConfig('SubcarrierSpacing', pSCS, 'NSizeGrid', pSizeGrid);
        [DLgrid, waveform, cInfo, pdschSymbols, trBlk, txInfo] = ...
            custom_downlink_tx(cCarrier, pSizeGrid, cfg.pDigitalModulation, pNCellID);
        pdschIndices = txInfo.pdschIndices;
        txSymbols = txInfo.pdschSymbols;
        berType = 'pre-FEC (custom, sem LDPC)';
    end
    waveform = waveform / max(abs(waveform));
    waveform = waveform - mean(waveform);
    waveform = 2 * waveform;

    % ---------------------------------------------------------------- canal
    waveform_pb = [];  uwoc_diag = [];  snr_rx_dB = NaN;  channelNotes = '';
    switch cfg.pChannel
        case 'software'
            offset = randi([0 cfg.pMaxRandomOffset]);
            rxClean = [zeros(offset, 1); waveform];
            sigPow_dBW = -10*log10(double(cInfo.Nfft));              % potencia NOMINAL assumida pelo awgn do original
            noise = complexNoise(size(rxClean), 10^((sigPow_dBW - cfg.pSNRdB)/10));
            rxWaveform = rxClean + noise;
            Ps = mean(abs(waveform).^2);  Pn = mean(abs(noise).^2);
            snr_rx_dB = 10*log10(Ps / Pn * cInfo.Nfft / K);          % SNR real nas subportadoras uteis
            t_off = (0:length(rxWaveform)-1)' / cInfo.SampleRate;    % equivalente ao comm.PhaseFrequencyOffset
            rxWaveform = rxWaveform .* exp(1i * (2*pi*cfg.pFrequencyOffset*t_off + (pi/180)*cfg.pPhaseOffset));
            channelNotes = sprintf('Canal RF em banda base: atraso %d amostras, AWGN nominal %.1f dB (SNR real nas subportadoras %.1f dB), CFO %.0f Hz, fase %.0f graus.', ...
                offset, cfg.pSNRdB, snr_rx_dB, cfg.pFrequencyOffset, cfg.pPhaseOffset);

        case 'b2b'
            [P1, Q1] = rat(cfg.pDACSampleRate / cInfo.SampleRate);
            waveform_up = resample(waveform, P1, Q1);
            if hasModDemod
                waveform_pb = modulate(real(waveform_up), cfg.pVLCCenterFrequency, cfg.pDACSampleRate, 'qam', imag(waveform_up));
                [rxReal, rxImag] = demod(waveform_pb(:), cfg.pVLCCenterFrequency, cfg.pDACSampleRate, 'qam');
            else
                waveform_pb = custom_modulate(real(waveform_up), cfg.pVLCCenterFrequency, cfg.pDACSampleRate, 'qam', imag(waveform_up));
                [rxReal, rxImag] = custom_demod(waveform_pb(:), cfg.pVLCCenterFrequency, cfg.pDACSampleRate, 'qam');
            end
            rxWaveform = resample(rxReal(:) + 1i*rxImag(:), Q1, P1);
            rxWaveform = rxWaveform(:);                              % sempre COLUNA
            snr_rx_dB = Inf;
            channelNotes = sprintf('Canal VLC ideal (b2b): Fs NR=%.2f MS/s -> DAC %.0f MS/s, portadora %.1f MHz, sem LED/UWOC/ruido.', ...
                cInfo.SampleRate/1e6, cfg.pDACSampleRate/1e6, cfg.pVLCCenterFrequency/1e6);

        case 'uwoc-b2b'
            p = cfg.uwoc;
            p.K = K;  p.SCS_Hz = pSCS * 1e3;
            if ~isfield(p, 'fc'),     p.fc = cfg.pVLCCenterFrequency; end
            if ~isfield(p, 'Fs_dac'), p.Fs_dac = cfg.pDACSampleRate;  end
            if isfield(p, 'SNR_ref') && strcmpi(p.SNR_ref, 'software'), p.SNR_dB = cfg.pSNRdB; end
            [rxWaveform, uwoc_diag] = uwoc_vlc_chain(waveform, cInfo.SampleRate, p);
            waveform_pb = uwoc_diag.waveform_pb;
            snr_rx_dB = uwoc_diag.SNR_rx_eff;
            up = uwoc_diag.params;
            channelNotes = sprintf('Canal UWOC: LED bias %.0f mA, swing +-%.0f mA, fotodetector %s, CIR %s, SNR eletrica %.1f dB (nas subportadoras %.1f dB), indice de modulacao rms %.2f.', ...
                up.I_bias, up.alpha*up.I_mod_A, up.pdLaw, up.cirMethod, uwoc_diag.SNR_elec_eff, snr_rx_dB, uwoc_diag.mod_index_rms);

        otherwise
            error('run_downlink:Canal', 'pChannel deve ser software, b2b ou uwoc-b2b.');
    end
    if any(~isfinite(rxWaveform))
        error('run_downlink:NonFinite', 'o canal %s produziu %d valores NaN/Inf (de %d); verifique a CIR e as normalizacoes.', ...
            cfg.pChannel, sum(~isfinite(rxWaveform)), numel(rxWaveform));
    end
    rxWaveform = rxWaveform / max(abs(rxWaveform));
    rxWaveform = rxWaveform - mean(rxWaveform);

    % ---------------------------------------------------------------- RX: sincronismo
    refGrid = DLgrid(:, 1:4);
    if useToolbox, [timingOffset, ~] = nrTimingEstimate(cCarrier, rxWaveform, refGrid);
    else,          [timingOffset, ~] = custom_nrTimingEstimate(cCarrier, rxWaveform, refGrid); end
    if timingOffset > length(rxWaveform)/2
        rxHalf = rxWaveform(1:floor(end/2));
        if useToolbox, [timingOffset, ~] = nrTimingEstimate(cCarrier, rxHalf, refGrid);
        else,          [timingOffset, ~] = custom_nrTimingEstimate(cCarrier, rxHalf, refGrid); end
    end
    rxWaveform_TC = rxWaveform(timingOffset+1 : timingOffset+length(waveform));

    freqCoarse = -20e3:1e3:20e3;  freqFine = -1000:5:1000;
    if useToolbox
        [corrected, cfoCoarse] = nrOFDM_CFO(rxWaveform_TC, freqCoarse, refGrid, cInfo.SampleRate, cCarrier);
        [rxWaveform_TCFC, cfoFine] = nrOFDM_CFO(corrected, freqFine, refGrid, cInfo.SampleRate, cCarrier);
    else
        [corrected, cfoCoarse] = custom_nrOFDM_CFO(rxWaveform_TC, freqCoarse, refGrid, cInfo.SampleRate, cCarrier);
        [rxWaveform_TCFC, cfoFine] = custom_nrOFDM_CFO(corrected, freqFine, refGrid, cInfo.SampleRate, cCarrier);
    end

    % ---------------------------------------------------------------- RX: demodulacao, equalizacao, metricas
    if useToolbox
        rxGrid = nrOFDMDemodulate(cCarrier, rxWaveform_TCFC, 'SampleRate', cInfo.SampleRate);
        rxGridPDSCH = rxGrid(:, 15:end);
        pdschDmrsIndices = nrPDSCHDMRSIndices(cCarrier, pdsch);
        pdschDmrsSymbols = nrPDSCHDMRS(cCarrier, pdsch);
        [hest, nVar] = nrChannelEstimate(rxGridPDSCH, pdschDmrsIndices, pdschDmrsSymbols);
        [pdschRxSym, pdschHest] = nrExtractResources(pdschIndices, rxGridPDSCH, hest);
        pdschEqSym = nrEqualizeMMSE(pdschRxSym, pdschHest, nVar);
        decodeDLSCH = nrDLSCHDecoder;
        decodeDLSCH.TargetCodeRate = cfg.pCodeRate;
        decodeDLSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
        decodeDLSCH.MaximumLDPCIterationCount = 6;
        [dlschLLRs, ~] = nrPDSCHDecode(cCarrier, pdsch, pdschEqSym, nVar);
        decodeDLSCH.TransportBlockLength = trBlkSizes;
        [decbits, blkerr] = decodeDLSCH(dlschLLRs, pdsch.Modulation, pdsch.NumLayers, 0);
        [num_erros, ber] = biterr(trBlk, decbits);
        evmFun = @(tx, rx) evmOnly(@evm, M, tx, rx);
    else
        rxGrid = custom_nrOFDMDemodulate(cCarrier, rxWaveform_TCFC);
        rxOpts = struct('plotSSS', false);
        evalc('[~, ~, pdschEqSym, ~, ~, ber, ~] = custom_downlink_rx(rxGrid, pSizeGrid, M, trBlk, pNCellID, txInfo, rxOpts);');
        num_erros = round(ber * numel(trBlk));
        blkerr = ber > 0;
        evmFun = @(tx, rx) evmOnly(@custom_evm, M, tx, rx);
    end

    rows = mod(pdschIndices(:) - 1, K) + 1;                          % subportadora de cada RE do PDSCH
    dcRow = K/2 + 1;
    all_ = true(size(rows));
    noDC = rows ~= dcRow;
    noDCnb = abs(rows - dcRow) > 2;
    evm_pct = evmFun(txSymbols, pdschEqSym);
    evm_noDC_pct = evmFun(txSymbols(noDC), pdschEqSym(noDC));
    evm_noDCnb_pct = evmFun(txSymbols(noDCnb), pdschEqSym(noDCnb));

    if cfg.verbose
        fprintf('  [%s | %s | %s] BER=%.4g  EVM=%.2f%% (sem DC %.2f%%)  SNR_rx=%.1f dB  timing=%d  CFO=%d%+d Hz\n', ...
            cfg.pChannel, cfg.pDigitalModulation, berType(1:min(end,7)), ber, evm_pct, evm_noDC_pct, snr_rx_dB, timingOffset, cfoCoarse, cfoFine);
    end

    R = struct('backend', ifelse(useToolbox, 'toolbox', 'custom'), 'pChannel', cfg.pChannel, ...
        'pDigitalModulation', cfg.pDigitalModulation, 'pModulationLevel', M, 'pSCS', pSCS, 'pSizeGrid', pSizeGrid, ...
        'ber', ber, 'num_errors', num_erros, 'blkerr', blkerr, 'berType', berType, ...
        'evm_pct', evm_pct, 'evm_noDC_pct', evm_noDC_pct, 'evm_noDCnb_pct', evm_noDCnb_pct, 'snr_rx_dB', snr_rx_dB, ...
        'timingOffset', timingOffset, 'cfo_coarse_hz', cfoCoarse, 'cfo_fine_hz', cfoFine, ...
        'sampleRate_nr', cInfo.SampleRate, 'bw_effective_hz', pSCS*1e3*K, 'channelNotes', channelNotes, ...
        'waveform', waveform, 'rxWaveform', rxWaveform_TCFC, 'DLgrid', DLgrid, 'cfg', cfg);
    if ~isempty(waveform_pb), R.waveform_pb = waveform_pb; end
    if ~isempty(uwoc_diag),   R.uwoc = uwoc_diag;          end
end

% ====================================================================== auxiliares
function cfg = fill_defaults(cfg)
    d = struct('backend', 'auto', 'pChannel', 'software', 'pDigitalModulation', '16QAM', 'pSCS', 15, 'pSizeGrid', 25, ...
        'pNCellID', 102, 'pCodeRate', 490/1024, 'pLayers', 1, 'pSNRdB', 50, 'pFrequencyOffset', 1.2e3, 'pPhaseOffset', 15, ...
        'pMaxRandomOffset', 100, 'pVLCCenterFrequency', 5e6, 'pDACSampleRate', 75e6, 'uwoc', struct(), 'seed', 42, 'verbose', false);
    fn = fieldnames(d);
    for i = 1:numel(fn)
        if ~isfield(cfg, fn{i}), cfg.(fn{i}) = d.(fn{i}); end
    end
end

function addLabPaths()
    here = fileparts(mfilename('fullpath'));
    cand = {'../../adalm2000', '../../adalm-pluto', '../../etc', '../../filters', '../../newradio'};
    for i = 1:numel(cand)
        d = fullfile(here, cand{i});
        if exist(d, 'dir') == 7, addpath(d); end
    end
end

function seedRandom(seed)
    if exist('rng', 'file') || exist('rng', 'builtin')
        rng(seed);
    else
        rand('seed', seed);  randn('seed', seed);                      %#ok<RAND>
    end
end

function n = complexNoise(sz, noisePower)
    % ruido gaussiano complexo de potencia TOTAL noisePower (mesma formula do awgn para sinal complexo)
    n = sqrt(noisePower/2) * (randn(sz) + 1i*randn(sz));
end

function e = evmOnly(f, M, tx, rx)
    [~, e] = f(M, tx, rx);
end

function v = ifelse(c, a, b)
    if c, v = a; else, v = b; end
end
