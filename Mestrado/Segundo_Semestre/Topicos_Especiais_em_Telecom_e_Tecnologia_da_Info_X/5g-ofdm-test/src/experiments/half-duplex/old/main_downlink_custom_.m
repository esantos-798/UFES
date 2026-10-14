% Test 5G New Radio over ADALM Pluto/2000
% Developed by Augusto Peterle / Refactored for Octave Compatibility
% 10/03/2023

clear;
close all;
clc;

% Add function paths
addpath('../../adalm2000');
addpath('../../adalm-pluto');
addpath('../../etc');
addpath('../../filters');
addpath('../../newradio');

%% Environment Detection
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
has5GToolbox = exist('nrOFDMModulate', 'file') > 0;

if isOctave
    try pkg load signal communications; catch; end
end

%% Initialization 
pSCS = 15;                      % Subcarrier spacing
pSizeGrid = 25;                 % Size grid - Number of resource blocks
pChannel = 'software';         % Channel can be: 'software', 'adalm-pluto', 'adalm2000', 'b2b'
pBW = 5e6;                      % Output element Bandwidth
pLayers = 1;                    % Number of layers send over the antenna
pDigitalModulation = '16QAM';  % Digital modulation: 'qam' or 'psk'
pModulationLevel = 16;         % Modulation M-QAM or M-PSK
pCyclicPrefix = 'Normal';       % Cyclic Prefix 'Normal' or 'Extended'
pNCellID = 102;                 % Number of Cell ID used in PBCH
pCodeRate = 490/1024;           % Code Rate

% Software Simulation parameters
pSNRdB = 50;                    % SNR for AWGN in dBrefGrid
pFrequencyOffset = 1.2e3;       % Frequency offset (Hz)
pPhaseOffset = 15;              % Phase offset (Degrees)

% ADALM PLUTO Parameters
pGain = -30;                    % Transmitter Gain in dB
pCenterFrequency = 2e9;         % Pass-band Center Frequency
pRadioIDTx = 'usb:0';           % Transmitter Radio ID
pRadioIDRx = 'usb:0';           % Receiver Radio ID

% ADALM 2000 Parameters
pVLCCenterFrequency = 5e6;      % VLC Center Frequency
pDACSampleRate = 75e6;          % DAC Available Sample Rate
pADCSampleRate = 100e6;         % ADC Available Sample Rate
pVLCUseCustomFilter = false;    % Add custom filter to VLC transmission

%% Initialization Calcs

% Calculate Effective Bandwidth
cBW = pSCS*1e3*12*pSizeGrid;    

% Init 5GNR Carrier
if has5GToolbox
    cCarrier = nrCarrierConfig('SubcarrierSpacing',pSCS,'NSizeGrid',pSizeGrid);
    cInfo = nrOFDMInfo(cCarrier);
else
    cCarrier = custom_nrCarrierConfig('SubcarrierSpacing',pSCS,'NSizeGrid',pSizeGrid);
    cInfo = custom_nrOFDMInfo(cCarrier);
end

cSampleRate = cInfo.SampleRate;

%% Downlink Transmission

if has5GToolbox
    % Create Synchronization Signal Block (SSB) (PSS + PBCH + SSS)
    ssbBlockGrid = initSSB(pNCellID,cCarrier,false);

    % Create Physical Downlink Shared Channel Configuration Instance
    pdsch = nrPDSCHConfig('NumLayers',pLayers,'PRBSet',[0:1:pSizeGrid-1]...
        ,'Modulation',pDigitalModulation);

    % Get Available Indices of PDSCH and PDSCH Info
    [pdschIndices,pdschInfo] = nrPDSCHIndices(cCarrier,pdsch);
    payloadLengthBits = pdschInfo.G;

    % Generate a DL-SCH object and the Transport Block Size
    [encodeDLSCH,trBlkSizes] = initDLSCH(pCodeRate,pdsch,cCarrier);

    % New Random Transport Block
    trBlk = randi([0 1],trBlkSizes,1);
    setTransportBlock(encodeDLSCH,trBlk);

    % Encode the Transport Block and map into digital symbols
    codedTrBlock = encodeDLSCH(pdsch.Modulation,1,pdschInfo.G,0);
    pdschSymbols = nrPDSCH(cCarrier,pdsch,codedTrBlock);

    % Map PDSCH symbols to a resource Grid
    pdschGrid = mapPDSCH(cCarrier,pdsch,pdschSymbols,...
        pdschIndices,pLayers,false);

    % Create PDCCH Physical Channel
    pdcch = nrPDCCHConfig('NStartBWP',0,'NSizeBWP',25);
    pdcch.AggregationLevel = 4;
    pdcch.CORESET.Duration = 1;
    pdcch.CORESET.FrequencyResources = [1,1,1,1];

    % Create a DCI (Downlink Control Information) word
    K = 64;             % Number of DCI message bits
    dciBits = randi([0 1],K,1,'int8');
    pdcchGrid = mapPDCCH(cCarrier,pdcch,dciBits,false);

    % Concatenate DL Resource Grid (SSB Grid + PDSCH Grid)
    DLgrid = mapDLGrid(ssbBlockGrid,pdschGrid,pdcchGrid,...
        cCarrier, false);

    % OFDM 
    [waveform,cInfo] = nrOFDMModulate(cCarrier,DLgrid,'Windowing',0);
else
    % Emulação de Downlink via Função Customizada
    [DLgrid, waveform, cInfo, pdschSymbols, trBlk] = custom_downlink_tx(cCarrier, pSizeGrid, pDigitalModulation);
end

% Normalization
waveform = waveform/(max(abs(waveform)));
waveform = waveform - mean(waveform);
waveform = 2*waveform;


%% Transmission Channel

if strcmp(pChannel,'software')
    %% Software simulation
    % Add random offset with noise
    offset = randi([0 1e2]); 
    rxWaveform = [zeros(offset,1);waveform];
    
    % Add white Gaussian noise to the waveform.
    rxWaveform = awgn(rxWaveform,pSNRdB,-10*log10(double(cInfo.Nfft)));
    
    % Introduce phase and frequency offsets
    if has5GToolbox
        PFO = comm.PhaseFrequencyOffset( ...
            'PhaseOffset',     pPhaseOffset, ...
            'FrequencyOffset', pFrequencyOffset,...
            'SampleRate',      cInfo.SampleRate);
        rxWaveform = PFO(rxWaveform);
    else
        t_offset = (0:length(rxWaveform)-1)' / cInfo.SampleRate;
        rxWaveform = rxWaveform .* exp(1i * (2*pi*pFrequencyOffset*t_offset + (pi/180)*pPhaseOffset));
    end
    
elseif strcmp(pChannel,'adalm-pluto')
    %% RF Channel - ADALM PLUTO
    load plutoTxFilter
    txPluto = sdrtx('Pluto','RadioID',pRadioIDTx,...
              'CenterFrequency',pCenterFrequency, ...
              'BasebandSampleRate',cInfo.SampleRate,'Gain',pGain,...
              'ShowAdvancedProperties',true,...
              'UseCustomFilter',true,filtTxConfig{:});
   
    load plutoRxFilter
    rxPluto = sdrrx('Pluto',...
           'RadioID',pRadioIDRx,...
           'CenterFrequency',pCenterFrequency,...
           'BasebandSampleRate',cInfo.SampleRate,...
           'SamplesPerFrame', 2*length(waveform),...
           'OutputDataType','double',...
           'UseCustomFilter',true,filtRxConfig{:}); 
    
    txPluto.transmitRepeat(waveform);
    rxWaveform = rxPluto();
    
elseif strcmp(pChannel,'adalm2000')
    %% VLC Channel with ADALM 2000
    [P1bb, Q1bb] = rat(pDACSampleRate/cInfo.SampleRate);
    waveform_up = resample(waveform,P1bb,Q1bb);
    
    waveform_pb = modulate(real(waveform_up),...
        pVLCCenterFrequency,...
        pDACSampleRate,'qam',imag(waveform_up));
   
    configM2k;
    
    aout.pushInterleaved(waveform_pb,1);
    rxWaveform_pb = ain.getSamplesInterleaved_matlab(5*length(waveform_pb));
    rxWaveform_pb = rxWaveform_pb.double;

    rxWaveform_pb_100 =  rxWaveform_pb ;
    
    rxWaveform_pb = rxWaveform_pb/max(abs(rxWaveform_pb));
    rxWaveform_pb = rxWaveform_pb - mean(rxWaveform_pb);
    
    [P1, Q1] = rat(pADCSampleRate/(pDACSampleRate/2));
    rxWaveform_pb = resample(rxWaveform_pb,Q1,P1);
    rxWaveform_pb_75 = rxWaveform_pb;
    
    clib.libm2k.libm2k.context.contextCloseAll();
    
    [rxReal,rxImag] = demod(rxWaveform_pb,pVLCCenterFrequency,pDACSampleRate,'qam');
    rxWaveform_up = rxReal + 1i.*rxImag;
    
    [P1bb2, Q1bb2] = rat((pADCSampleRate)/(cInfo.SampleRate));
    rxWaveform = resample(rxWaveform_up,Q1bb,P1bb);
    rxWaveform = rxWaveform.';
else
    %% B2B Pass band simulation
    [P1bb, Q1bb] = rat(pDACSampleRate/cInfo.SampleRate);
    waveform_up = resample(waveform,P1bb,Q1bb);
    
    waveform_pb = modulate(real(waveform_up),...
        pVLCCenterFrequency,...
        pDACSampleRate,'qam',imag(waveform_up));
    
    rxWaveform_pb = waveform_pb.';
    
    [rxReal,rxImag] = demod(rxWaveform_pb,...
        pVLCCenterFrequency,pDACSampleRate,'qam');
    rxWaveform_up = rxReal + 1i.*rxImag;
    
    rxWaveform = resample(rxWaveform_up,Q1bb,P1bb);
    rxWaveform = rxWaveform.';
end

% Reception Normalization
rxWaveform = rxWaveform/max(abs(rxWaveform));
rxWaveform = rxWaveform - mean(rxWaveform);

%% Reception

refGrid = DLgrid(:,1:4);

% Timing estimation
if has5GToolbox
    [timingOffset,mag] = nrTimingEstimate(cCarrier,rxWaveform,refGrid);
else
    [timingOffset,mag] = custom_nrTimingEstimate(cCarrier,rxWaveform,refGrid);
end

if timingOffset > length(rxWaveform)/2
    rxWaveform_half = rxWaveform(1:end/2);
    if has5GToolbox
        [timingOffset,mag] = nrTimingEstimate(cCarrier,rxWaveform_half,refGrid);
    else
        [timingOffset,mag] = custom_nrTimingEstimate(cCarrier,rxWaveform_half,refGrid);
    end
end

% Time offset Correction
rxWaveform_TC = rxWaveform(timingOffset+1:timingOffset+length(waveform));
fprintf('Timing Offset Sample: %d\n',timingOffset);

% Coarse Frequency Offset Correction 
frequencyCorrectionRange = -20e3:1e3:20e3;
if has5GToolbox
    [correctedWaveform,appliedFrequencyCorrection] = nrOFDM_CFO(rxWaveform_TC,...
        frequencyCorrectionRange,refGrid,cInfo.SampleRate,cCarrier);
else
    [correctedWaveform,appliedFrequencyCorrection] = custom_nrOFDM_CFO(rxWaveform_TC,...
        frequencyCorrectionRange,refGrid,cInfo.SampleRate,cCarrier);
end
fprintf(' Coarse Frequency Correction : %d\n',appliedFrequencyCorrection);

% Fine Frequency Offset Correction
frequencyCorrectionRange = -1000:5:1000;
if has5GToolbox
    [rxWaveform_TCFC,appliedFrequencyCorrection] = nrOFDM_CFO(correctedWaveform,...
        frequencyCorrectionRange,refGrid,cInfo.SampleRate,cCarrier);
else
    [rxWaveform_TCFC,appliedFrequencyCorrection] = custom_nrOFDM_CFO(correctedWaveform,...
        frequencyCorrectionRange,refGrid,cInfo.SampleRate,cCarrier);
end
fprintf(' Fine Frequency Correction : %d\n',appliedFrequencyCorrection);

% OFDM Demodulate
if strcmp(pChannel,'adalm-pluto')
    if strcmp(txPluto.RadioID,rxPluto.RadioID)
        if has5GToolbox
            rxGrid = nrOFDMDemodulate(cCarrier,rxWaveform_TC,'SampleRate',cInfo.SampleRate);
        else
            rxGrid = custom_nrOFDMDemodulate(cCarrier,rxWaveform_TC);
        end
    end
else
    if has5GToolbox
        rxGrid = nrOFDMDemodulate(cCarrier,rxWaveform_TCFC,'SampleRate',cInfo.SampleRate);
    else
        rxGrid = custom_nrOFDMDemodulate(cCarrier,rxWaveform_TCFC);
    end
end

% RX Demodulation & Decoding Process
if has5GToolbox
    % Detect PSS NID2
    NID2 = detectPSS(rxGrid, pSizeGrid);

    % Extract SSS and detect Cell
    pbCHoffset = round((pSizeGrid*12 - 240)/2);
    startSSS = pbCHoffset + 57;
    sssRx = rxGrid(startSSS:startSSS+126, 4);
    [NID1, rxCellid] = detecSSS(sssRx, NID2, true);

    % Decode PDSCH
    rxGridPDSCH = rxGrid(:, 15:end);
    pdschDmrsIndices = nrPDSCHDMRSIndices(cCarrier, pdsch);
    pdschDmrsSymbols = nrPDSCHDMRS(cCarrier, pdsch);

    % Channel Estimation
    [hest, nVar, pdschHestInfo] = nrChannelEstimate(rxGridPDSCH, pdschDmrsIndices, pdschDmrsSymbols);

    % Get PDSCH Indices and symbols
    [pdschIndices, pdschIndicesInfo] = nrPDSCHIndices(cCarrier, pdsch);
    [pdschRxSym, pdschHest] = nrExtractResources(pdschIndices, rxGridPDSCH, hest);

    % Apply Equalizer
    pdschEqSym = nrEqualizeMMSE(pdschRxSym, pdschHest, nVar);

    % Recover bits
    decodeDLSCH = nrDLSCHDecoder;
    decodeDLSCH.MultipleHARQProcesses = false;
    decodeDLSCH.TargetCodeRate = pCodeRate;
    decodeDLSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
    decodeDLSCH.MaximumLDPCIterationCount = 6;

    [dlschLLRs, rxSymbols] = nrPDSCHDecode(cCarrier, pdsch, pdschEqSym, nVar);
    decodeDLSCH.TransportBlockLength = trBlkSizes;
    [decbits, blkerr] = decodeDLSCH(dlschLLRs, pdsch.Modulation, pdsch.NumLayers, 0);

    % Bit Error Rate & EVM
    [num_erros, ber] = biterr(trBlk, decbits);
    fprintf('BER = %.4f, Errors = %d\n', ber, num_erros);

    [EVM_dB, EVM_porc] = evm(pModulationLevel, pdschSymbols, pdschEqSym);
    fprintf('EVM Percentual = %.2f%%\n', EVM_porc);
    
    % Variáveis para plotagem no MATLAB
    rxGridTC = nrOFDMDemodulate(cCarrier, rxWaveform_TC);
    rxGridPDSCH_TC = rxGridTC(:, 15:end);
    [pdschRxSymTC, ~] = nrExtractResources(pdschIndices, rxGridPDSCH_TC);
    txSymbols = pdschGrid(pdschIndices);
else
    % Execução via função customizada no Octave
    [rxCellid, pdschRxSym, pdschEqSym, hest, decbits, ber, EVM_porc] = ...
        custom_downlink_rx(rxGrid, DLgrid, pModulationLevel, trBlk, pNCellID);
        
    % Variáveis para plotagem no Octave
    rxGridTC = custom_nrOFDMDemodulate(cCarrier, rxWaveform_TC);
    pdschRxSymTC = rxGridTC(:);
    txSymbols = DLgrid(:);
end

%% Plot Figures

%% Show TX/RX spectra
figure();
subplot(2,1,1);
if ~strcmp(pChannel,'adalm2000')
    [Sinal_ff,~,f,df] = FFT_pot2(waveform.',1/cInfo.SampleRate);
else
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(waveform_pb.',1/pDACSampleRate);
end
espectroAbs = mag2db(abs(Sinal_ff));
plot(f,fftshift(espectroAbs));
grid on;
title('5G NR OFDM Baseband Tx Spectrum - SSB + PDSCH');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (Hz)');
ylim([-200,-100]);


subplot(2,1,2);
if ~strcmp(pChannel,'adalm2000')
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(rxWaveform.',1/cInfo.SampleRate);
else
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(rxWaveform_pb,1/(pDACSampleRate));
end
espectroAbs = mag2db(abs(Sinal_ff));
plot(f,fftshift(espectroAbs));
grid on;
title('5G NR OFDM Baseband RX Spectrum - SSB + PDSCH');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (Hz)');
ylim([-200,-100]);
drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'figura2.png');
fprintf('Figura salva: figura2.png\n');

%% Resource Grid Plot
figure();
imagesc(abs(DLgrid));
axis xy;
xlabel('OFDM symbol');
ylabel('Subcarrier');
title('5G NR Resource Grid - SSB + PDCCH + PDSCH');

drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'figura3.png');
fprintf('Figura salva: figura3.png\n');

%% Time Domain Plot
figure();
plot(real(waveform));
hold on;
plot(real(rxWaveform_TCFC));
plot(real(rxWaveform_TC));
legend('TX', 'RX\_TCFC', 'RX\_TC');

drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'figura4.png');
fprintf('Figura salva: figura4.png\n');

%% Constellation Diagram Plot
figure('Name', 'Diagrama de Constelação');
a = max(abs(real(txSymbols)));
if isempty(a) || a == 0
    a = 1.5; 
end

subplot(2,2,1);
plot(real(txSymbols),imag(txSymbols),'r.');
title ('Const TX');
axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
xlabel('Real'), ylabel('Imag'), grid, hold on;

subplot(2,2,2);
plot(real(pdschRxSymTC),imag(pdschRxSymTC),'k.');
title ('Const RX with Time Sync');
axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
xlabel('Real'), ylabel('Imag'), grid, hold on;

subplot(2,2,3);
plot(real(pdschRxSym),imag(pdschRxSym),'g.');
title ('Const RX with Time and Freq Sync');
axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
xlabel('Real'), ylabel('Imag'), grid, hold on;

subplot(2,2,4);
plot(real(pdschEqSym),imag(pdschEqSym),'b.');
title ('Const RX Time and Freq Sync + Eq');
axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
xlabel('Real'), ylabel('Imag'), grid, hold on;

drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'figura5.png');
fprintf('Figura salva: figura5.png\n');

%% Estimated Channel
figure();
mesh(abs(hest(:,:,1,1)));
title('Channel Estimate');
xlabel('OFDM Symbol');
ylabel("Subcarrier");
zlabel("Magnitude");

%% Debug ADALM 2000 spectra
if strcmp(pChannel,'adalm2000') || strcmp(pChannel,'b2b')
    figure();
    subplot(3,1,1);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(rxWaveform_pb,1/(pDACSampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f/1e6,fftshift(espectroAbs));
    grid on;
    title('b.3) Passband Spectrum Fs2 = 75 MSPS');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (MHz)');
    ylim([-200,-100]);
    
    subplot(3,1,2);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(rxWaveform_up,1/(pDACSampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f/1e6,fftshift(espectroAbs));
    grid on;
    title('b.2) Baseband Upsampled Spectrum Fs2 = 75MSPS');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (MHz)');
    ylim([-200,-100]);
    
    subplot(3,1,3);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(rxWaveform.',1/(cInfo.SampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f/1e6,fftshift(espectroAbs));
    grid on;
    title('b.1) Baseband Spectrum Fs1 = 7.68 MSPS');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (MHz)');
    ylim([-200,-100]);
    
    figure();
    subplot(3,1,3);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(waveform_pb.',1/(pDACSampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f/1e6,fftshift(espectroAbs));
    grid on;
    title('a.3) Passband Spectrum Fs3 = 75 MSPS');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (MHz)');
    ylim([-200,-100]);
    
    subplot(3,1,2);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(waveform_up.',1/(pDACSampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f/1e6,fftshift(espectroAbs));
    grid on;
    title('a.2) Baseband Upsampled Spectrum Fs2 = 75MSPS');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (MHz)');
    ylim([-200,-100]);
    
    subplot(3,1,1);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(waveform.',1/(cInfo.SampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f/1e6,fftshift(espectroAbs));
    grid on;
    title('a.1) Baseband Spectrum Fs1 = 7.68MSPS');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (MHz)');
    ylim([-200,-100]);
end