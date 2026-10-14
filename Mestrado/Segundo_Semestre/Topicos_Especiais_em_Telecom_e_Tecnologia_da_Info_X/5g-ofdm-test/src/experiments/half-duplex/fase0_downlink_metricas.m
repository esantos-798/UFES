function R = fase0_downlink_metricas(cfg)
%FASE0_DOWNLINK_METRICAS Uma execução do downlink (TX + canal + RX + métricas).
%
%   R = fase0_downlink_metricas(cfg)  — struct com BER, EVM, offsets, etc.
%
%   Campos úteis de cfg:
%     pChannel          — 'software' | 'b2b' | 'uwoc-b2b'  (cfg.uwoc = struct parcial, ver uwoc_defaults)
%     pDigitalModulation — 'QPSK' | '16QAM' | '64QAM'
%     pModulationLevel  — 4 | 16 | 64
%     pSNRdB            — default 50 (software)
%     pFrequencyOffset  — default 1.2e3 Hz (software)
%     pPhaseOffset      — default 15 deg (software)
%     seed              — default 42 (reprodutibilidade)
%     verbose           — default false

    if nargin < 1
        cfg = struct();
    end
    cfg = fill_defaults(cfg);

    rng(cfg.seed);

    has5GToolbox = exist('nrOFDMModulate', 'file') > 0;
    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    if isOctave
        try
            pkg load signal communications;
        catch
        end
    end

    scriptDir = fileparts(mfilename('fullpath'));
    addpath(fullfile(scriptDir, '../../adalm2000'));
    addpath(fullfile(scriptDir, '../../adalm-pluto'));
    addpath(fullfile(scriptDir, '../../etc'));
    addpath(fullfile(scriptDir, '../../filters'));
    addpath(fullfile(scriptDir, '../../newradio'));

    pSCS = 15;
    pSizeGrid = 25;
    pLayers = 1;
    pNCellID = 102;
    pCodeRate = 490/1024;
    pVLCCenterFrequency = 5e6;
    pDACSampleRate = 75e6;

    if has5GToolbox
        cCarrier = nrCarrierConfig('SubcarrierSpacing', pSCS, 'NSizeGrid', pSizeGrid);
        cInfo = nrOFDMInfo(cCarrier);
    else
        cCarrier = custom_nrCarrierConfig('SubcarrierSpacing', pSCS, 'NSizeGrid', pSizeGrid);
        cInfo = custom_nrOFDMInfo(cCarrier);
    end

    %% --- TX (igual ao main_downlink) ---
    if has5GToolbox
        ssbBlockGrid = initSSB(pNCellID, cCarrier, false);
        pdsch = nrPDSCHConfig('NumLayers', pLayers, 'PRBSet', 0:(pSizeGrid-1), ...
            'Modulation', cfg.pDigitalModulation);
        [pdschIndices, pdschInfo] = nrPDSCHIndices(cCarrier, pdsch);
        [encodeDLSCH, trBlkSizes] = initDLSCH(pCodeRate, pdsch, cCarrier);
        trBlk = randi([0 1], trBlkSizes, 1);
        setTransportBlock(encodeDLSCH, trBlk);
        codedTrBlock = encodeDLSCH(pdsch.Modulation, 1, pdschInfo.G, 0);
        pdschSymbols = nrPDSCH(cCarrier, pdsch, codedTrBlock);
        pdschGrid = mapPDSCH(cCarrier, pdsch, pdschSymbols, pdschIndices, pLayers, false);
        pdcch = nrPDCCHConfig('NStartBWP', 0, 'NSizeBWP', 25);
        pdcch.AggregationLevel = 4;
        pdcch.CORESET.Duration = 1;
        pdcch.CORESET.FrequencyResources = [1, 1, 1, 1];
        dciBits = randi([0 1], 64, 1, 'int8');
        pdcchGrid = mapPDCCH(cCarrier, pdcch, dciBits, false);
        DLgrid = mapDLGrid(ssbBlockGrid, pdschGrid, pdcchGrid, cCarrier, false);
        [waveform, cInfo] = nrOFDMModulate(cCarrier, DLgrid, 'Windowing', 0);
        txInfo = [];
    else
        [DLgrid, waveform, cInfo, pdschSymbols, trBlk, txInfo] = ...
            custom_downlink_tx(cCarrier, pSizeGrid, cfg.pDigitalModulation, pNCellID);
        pdschIndices = txInfo.pdschIndices;
        trBlkSizes = numel(trBlk);
        pdsch = [];
    end

    waveform = waveform / max(abs(waveform));
    waveform = waveform - mean(waveform);
    waveform = 2 * waveform;

    %% --- Canal ---
    channelNotes = '';
    rxWaveform_pb = [];
    waveform_up = [];
    waveform_pb = [];
    uwoc_diag = [];

    switch cfg.pChannel
        case 'software'
            offset = randi([0 100]);
            rxWaveform = [zeros(offset, 1); waveform];
            if has5GToolbox
                rxWaveform = awgn(rxWaveform, cfg.pSNRdB, -10*log10(double(cInfo.Nfft)));
                PFO = comm.PhaseFrequencyOffset( ...
                    'PhaseOffset', cfg.pPhaseOffset, ...
                    'FrequencyOffset', cfg.pFrequencyOffset, ...
                    'SampleRate', cInfo.SampleRate);
                rxWaveform = PFO(rxWaveform);
            else
                rxWaveform = custom_awgn(rxWaveform, cfg.pSNRdB, -10*log10(double(cInfo.Nfft)));
                t_offset = (0:length(rxWaveform)-1)' / cInfo.SampleRate;
                rxWaveform = rxWaveform .* exp(1i * (2*pi*cfg.pFrequencyOffset*t_offset + (pi/180)*cfg.pPhaseOffset));
            end
            channelNotes = sprintf(['Canal RF em banda base: atraso %d amostras, AWGN %.1f dB, ', ...
                'CFO %.0f Hz, fase %.0f°.'], offset, cfg.pSNRdB, cfg.pFrequencyOffset, cfg.pPhaseOffset);

        case 'b2b'
            [P1bb, Q1bb] = rat(pDACSampleRate / cInfo.SampleRate);
            waveform_up = resample(waveform, P1bb, Q1bb);
            waveform_pb = fase0_apply_duc(waveform_up, pVLCCenterFrequency, pDACSampleRate);
            rxWaveform_pb = waveform_pb(:);
            [rxReal, rxImag] = fase0_apply_ddc(rxWaveform_pb, pVLCCenterFrequency, pDACSampleRate);
            rxWaveform_up = rxReal + 1i .* rxImag;
            rxWaveform = resample(rxWaveform_up, Q1bb, P1bb);
            rxWaveform = rxWaveform(:);   % coluna: rxWaveform.' gerava LINHA e custom_nrOFDM_CFO expandia para N x N (estouro de memoria)
            channelNotes = sprintf(['Canal VLC ideal (b2b): Fs NR=%.2f MS/s -> DAC %.0f MS/s, ', ...
                'portadora %.1f MHz, sem LED/UWOC/ruído.'], cInfo.SampleRate/1e6, pDACSampleRate/1e6, pVLCCenterFrequency/1e6);

        case 'uwoc-b2b'
            % Fase 1: DUC -> LED -> agua (Double Gamma) -> fotodetector -> DDC (ver uwoc_vlc_chain)
            [rxWaveform, uwoc_diag] = uwoc_vlc_chain(waveform, cInfo.SampleRate, cfg.uwoc);
            waveform_pb = uwoc_diag.waveform_pb;
            up = uwoc_diag.params;
            channelNotes = sprintf(['Canal UWOC (Fase 1): LED bias %.0f mA, swing +-%.0f mA, fotodetector %s, ', ...
                'CIR DGF (%s), SNR eletrico %.1f dB, indice de modulacao rms %.2f.'], ...
                up.I_bias, up.alpha*up.I_mod_A, up.pdLaw, up.cirMethod, up.SNR_dB, uwoc_diag.mod_index_rms);

        otherwise
            error('fase0_downlink_metricas:Canal', 'Canais suportados: software, b2b, uwoc-b2b.');
    end

    rxWaveform = rxWaveform / max(abs(rxWaveform));
    rxWaveform = rxWaveform - mean(rxWaveform);

    %% --- RX (sincronismo + equalização + métricas) ---
    refGrid = DLgrid(:, 1:4);

    if has5GToolbox
        [timingOffset, ~] = nrTimingEstimate(cCarrier, rxWaveform, refGrid);
    else
        [timingOffset, ~] = custom_nrTimingEstimate(cCarrier, rxWaveform, refGrid);
    end
    if timingOffset > length(rxWaveform)/2
        rxHalf = rxWaveform(1:end/2);
        if has5GToolbox
            [timingOffset, ~] = nrTimingEstimate(cCarrier, rxHalf, refGrid);
        else
            [timingOffset, ~] = custom_nrTimingEstimate(cCarrier, rxHalf, refGrid);
        end
    end

    rxWaveform_TC = rxWaveform(timingOffset+1:timingOffset+length(waveform));

    freqCoarse = -20e3:1e3:20e3;
    if has5GToolbox
        [correctedWaveform, cfoCoarse] = nrOFDM_CFO(rxWaveform_TC, freqCoarse, refGrid, cInfo.SampleRate, cCarrier);
    else
        [correctedWaveform, cfoCoarse] = custom_nrOFDM_CFO(rxWaveform_TC, freqCoarse, refGrid, cInfo.SampleRate, cCarrier);
    end

    freqFine = -1000:5:1000;
    if has5GToolbox
        [rxWaveform_TCFC, cfoFine] = nrOFDM_CFO(correctedWaveform, freqFine, refGrid, cInfo.SampleRate, cCarrier);
    else
        [rxWaveform_TCFC, cfoFine] = custom_nrOFDM_CFO(correctedWaveform, freqFine, refGrid, cInfo.SampleRate, cCarrier);
    end

    if has5GToolbox
        rxGrid = nrOFDMDemodulate(cCarrier, rxWaveform_TCFC, 'SampleRate', cInfo.SampleRate);
        rxGridPDSCH = rxGrid(:, 15:end);
        pdschDmrsIndices = nrPDSCHDMRSIndices(cCarrier, pdsch);
        pdschDmrsSymbols = nrPDSCHDMRS(cCarrier, pdsch);
        [hest, nVar] = nrChannelEstimate(rxGridPDSCH, pdschDmrsIndices, pdschDmrsSymbols);
        [pdschIndices, ~] = nrPDSCHIndices(cCarrier, pdsch);
        [pdschRxSym, pdschHest] = nrExtractResources(pdschIndices, rxGridPDSCH, hest);
        pdschEqSym = nrEqualizeMMSE(pdschRxSym, pdschHest, nVar);
        decodeDLSCH = nrDLSCHDecoder;
        decodeDLSCH.TargetCodeRate = pCodeRate;
        decodeDLSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
        decodeDLSCH.MaximumLDPCIterationCount = 6;
        [dlschLLRs, ~] = nrPDSCHDecode(cCarrier, pdsch, pdschEqSym, nVar);
        decodeDLSCH.TransportBlockLength = trBlkSizes;
        [decbits, blkerr] = decodeDLSCH(dlschLLRs, pdsch.Modulation, pdsch.NumLayers, 0);
        [num_erros, ber] = biterr(trBlk, decbits);
        [~, EVM_porc] = evm(cfg.pModulationLevel, pdschSymbols, pdschEqSym);
        berType = 'pos-LDPC (5G Toolbox)';
    else
        rxGrid = custom_nrOFDMDemodulate(cCarrier, rxWaveform_TCFC);
        [~, pdschRxSym, pdschEqSym, hest, decbits, ber, EVM_porc] = ...
            custom_downlink_rx(rxGrid, pSizeGrid, cfg.pModulationLevel, trBlk, pNCellID, txInfo);
        num_erros = round(ber * numel(trBlk));
        blkerr = ber > 0;
        berType = 'pre-FEC (custom, sem LDPC)';
    end

    if cfg.verbose
        fprintf('  [%s | %s] BER=%.4g  EVM=%.2f%%  timing=%d  CFO=%d+%d Hz\n', ...
            cfg.pChannel, cfg.pDigitalModulation, ber, EVM_porc, timingOffset, cfoCoarse, cfoFine);
    end

    R = struct();
    R.pChannel = cfg.pChannel;
    R.pDigitalModulation = cfg.pDigitalModulation;
    R.pModulationLevel = cfg.pModulationLevel;
    R.ber = ber;
    R.num_errors = num_erros;
    R.blkerr = blkerr;
    R.evm_pct = EVM_porc;
    R.berType = berType;
    R.timingOffset = timingOffset;
    R.cfo_coarse_hz = cfoCoarse;
    R.cfo_fine_hz = cfoFine;
    R.sampleRate_nr = cInfo.SampleRate;
    R.bw_effective_hz = pSCS*1e3*12*pSizeGrid;
    R.channelNotes = channelNotes;
    R.has5GToolbox = has5GToolbox;
    R.waveform = waveform;
    R.rxWaveform = rxWaveform_TCFC;
    R.DLgrid = DLgrid;
    if ~isempty(waveform_pb)
        R.waveform_pb = waveform_pb;
    end
    if ~isempty(uwoc_diag)
        R.uwoc = uwoc_diag;
    end
end

function cfg = fill_defaults(cfg)
    if ~isfield(cfg, 'pChannel'), cfg.pChannel = 'software'; end
    if ~isfield(cfg, 'pDigitalModulation'), cfg.pDigitalModulation = '16QAM'; end
    if ~isfield(cfg, 'pModulationLevel'), cfg.pModulationLevel = 16; end
    if ~isfield(cfg, 'pSNRdB'), cfg.pSNRdB = 50; end
    if ~isfield(cfg, 'pFrequencyOffset'), cfg.pFrequencyOffset = 1.2e3; end
    if ~isfield(cfg, 'pPhaseOffset'), cfg.pPhaseOffset = 15; end
    if ~isfield(cfg, 'seed'), cfg.seed = 42; end
    if ~isfield(cfg, 'verbose'), cfg.verbose = false; end
    if ~isfield(cfg, 'uwoc'), cfg.uwoc = struct(); end
end

function pb = fase0_apply_duc(waveform_up, fc, Fs)
    if exist('modulate', 'file') == 2
        pb = modulate(real(waveform_up), fc, Fs, 'qam', imag(waveform_up));
    else
        pb = custom_modulate(real(waveform_up), fc, Fs, 'qam', imag(waveform_up));
    end
end

function [rxReal, rxImag] = fase0_apply_ddc(rxWaveform_pb, fc, Fs)
    if exist('demod', 'file') == 2
        [rxReal, rxImag] = demod(rxWaveform_pb, fc, Fs, 'qam');
    else
        [rxReal, rxImag] = custom_demod(rxWaveform_pb, fc, Fs, 'qam');
    end
end
