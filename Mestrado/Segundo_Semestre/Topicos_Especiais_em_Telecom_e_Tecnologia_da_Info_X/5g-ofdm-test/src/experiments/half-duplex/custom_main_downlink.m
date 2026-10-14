% Test 5G New Radio over ADALM Pluto/2000
% Developed by Augusto Peterle
% 10/03/2023

clear ;
close all;
clc;

% Add function paths
addpath('../../adalm2000');
addpath('../../adalm-pluto');
addpath('../../etc');
addpath('../../filters');
addpath('../../newradio');

%% Environment Detection (Octave / MATLAB sem 5G Toolbox)
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
has5GToolbox = exist('nrOFDMModulate', 'file') > 0;
hasModDemod = exist('modulate', 'file') > 0 && exist('demod', 'file') > 0;   % Signal Processing Toolbox (Octave nao tem)

if isOctave
    try pkg load signal communications; catch; end
end

%% Initialization 
pSCS = 15;                      % Subcarrier spacing
pSizeGrid = 25;                 % Size grid - Number of resource blocks
pChannel = 'software';         % Channel can be: 'software', 'adalm-pluto', 'adalm2000', 'b2b'
% pChannel = 'uwoc-b2b' : enlace VLC/UWOC (DUC -> LED -> agua Double Gamma -> fotodetector -> DDC), ver uwoc_vlc_chain.m
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

% UWOC parameters (only for pChannel = 'uwoc-b2b'; omitted fields use uwoc_defaults.m)
pUWOC = struct('SNR_dB', 30, ...            % SNR (dB), Inf = sem ruido; referencia em SNR_ref
               'SNR_ref', 'electrical', ... % 'electrical' | 'rx' (subportadoras uteis) | 'software' (usa pSNRdB)
               'pdLaw', 'linear', ...       % fotodetector: 'linear' | 'square' (replica SimLED_conv.m)
               'I_mod_A', 50);              % pico do sinal modulante (mA); excursao = alpha*I_mod_A
% fc e Fs_dac do UWOC seguem pVLCCenterFrequency e pDACSampleRate (ajuste-os ao mudar pSCS)
pADCSampleRate = 100e6;         % ADC Available Sample Rate
pVLCUseCustomFilter = false;    % Add custom filter to VLC transmission

%% Initialization Calcs

% Calculate Effective Bandwidth
cBW = pSCS*1e3*12*pSizeGrid;    

% Init 5GNR Carrier
if has5GToolbox
    cCarrier = nrCarrierConfig('SubcarrierSpacing',pSCS,'NSizeGrid',pSizeGrid);
else
    cCarrier = custom_nrCarrierConfig('SubcarrierSpacing',pSCS,'NSizeGrid',pSizeGrid);
end

% Get OFDM Info
if has5GToolbox
    cInfo = nrOFDMInfo(cCarrier);
else
    cInfo = custom_nrOFDMInfo(cCarrier);
end

% SampleRate
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
    % Emulacao de Downlink via funcoes custom (SSB + PDCCH + PDSCH/DMRS, sem LDPC)
    [DLgrid, waveform, cInfo, pdschSymbols, trBlk, txInfo] = ...
        custom_downlink_tx(cCarrier, pSizeGrid, pDigitalModulation, pNCellID);
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
    
    % Add white Gaussian noise to the waveform. Note that the SNR only
    rxWaveform = awgn(rxWaveform,pSNRdB,-10*log10(double(cInfo.Nfft)));
    
if has5GToolbox
        % Introduce phase and frequency offsets
        PFO = comm.PhaseFrequencyOffset( ...
        'PhaseOffset',     pPhaseOffset, ...
        'FrequencyOffset', pFrequencyOffset,...
        'SampleRate',      cInfo.SampleRate);
    
        % Insert Phase and Frequency Offset
        rxWaveform = PFO(rxWaveform);
else
    % Introduce phase and frequency offsets (equivalente ao comm.PhaseFrequencyOffset)
    t_offset = (0:length(rxWaveform)-1)' / cInfo.SampleRate;
    rxWaveform = rxWaveform .* exp(1i * (2*pi*pFrequencyOffset*t_offset + (pi/180)*pPhaseOffset));
end
    
elseif strcmp(pChannel,'adalm-pluto')
    %% RF Channel - ADALM PLUTO
    % Create a tx pluto object
    load plutoTxFilter
    txPluto = sdrtx('Pluto','RadioID',pRadioIDTx,...
              'CenterFrequency',pCenterFrequency, ...
              'BasebandSampleRate',cInfo.SampleRate,'Gain',pGain,...
              'ShowAdvancedProperties',true,...
              'UseCustomFilter',true,filtTxConfig{:});
   
    % Create a rx pluto object
    load plutoRxFilter
    rxPluto = sdrrx('Pluto',...
           'RadioID',pRadioIDRx,...
           'CenterFrequency',pCenterFrequency,...
           'BasebandSampleRate',cInfo.SampleRate,...
           'SamplesPerFrame', 2*length(waveform),...
           'OutputDataType','double',...
           'UseCustomFilter',true,filtRxConfig{:}); 
    
    % Transmit pluto Data
    % This run in a separate embedded thread
    txPluto.transmitRepeat(waveform);
    
    % Receive a ADALM Pluto Frame
    rxWaveform = rxPluto();
    
elseif strcmp(pChannel,'adalm2000')
    %% VLC Channel with ADALM 2000
    
    % Upsample the baseband OFDM symbol to the DAC Sample Rate
    [P1bb, Q1bb] = rat(pDACSampleRate/cInfo.SampleRate);
    waveform_up = resample(waveform,P1bb,Q1bb);
    
    % Apply DUC - Digital Upconverter - Analogical QAM
if hasModDemod
        waveform_pb = modulate(real(waveform_up),...
            pVLCCenterFrequency,...
            pDACSampleRate,'qam',imag(waveform_up));
else
    waveform_pb = custom_modulate(real(waveform_up),...
        pVLCCenterFrequency,...
        pDACSampleRate,'qam',imag(waveform_up));
end
   
    % Configure ADALM2000 Driver
    configM2k;
    
    % Write in ADALM 2000 in channel 1
    aout.pushInterleaved(waveform_pb,1);
    rxWaveform_pb = ain.getSamplesInterleaved_matlab(5*length(waveform_pb));
    rxWaveform_pb = rxWaveform_pb.double;

    rxWaveform_pb_100 =  rxWaveform_pb ;
    
    % Normalize and remove DC of the received signal
    rxWaveform_pb = rxWaveform_pb/max(abs(rxWaveform_pb));
    rxWaveform_pb = rxWaveform_pb - mean(rxWaveform_pb);
    
    % Resample the received signal to match ADC to DAC Sampling Rate
    [P1, Q1] = rat(pADCSampleRate/(pDACSampleRate/2));
    rxWaveform_pb = resample(rxWaveform_pb,Q1,P1);
    rxWaveform_pb_75 = rxWaveform_pb;
    
    % Clear context
    clib.libm2k.libm2k.context.contextCloseAll();
    
    % Demodulate Back to the Baseband
if hasModDemod
        [rxReal,rxImag] = demod(rxWaveform_pb,pVLCCenterFrequency,pDACSampleRate,'qam');
else
    [rxReal,rxImag] = custom_demod(rxWaveform_pb,pVLCCenterFrequency,pDACSampleRate,'qam');
end
    rxWaveform_up = rxReal + 1i.*rxImag;
    
    % Downsample
    [P1bb2, Q1bb2] = rat((pADCSampleRate)/(cInfo.SampleRate));
    rxWaveform = resample(rxWaveform_up,Q1bb,P1bb);
    
    % Convert to column vector
    rxWaveform = rxWaveform.';
elseif strcmp(pChannel,'uwoc-b2b')
    %% UWOC/VLC Pass band simulation: DUC -> LED -> water (Double Gamma) -> photodetector -> DDC
    % Link UWOC parameters to the main parameters (fields already set in pUWOC take precedence)
    pUWOC.K = pSizeGrid*12;  pUWOC.SCS_Hz = pSCS*1e3;
    if ~isfield(pUWOC,'fc'),     pUWOC.fc = pVLCCenterFrequency; end
    if ~isfield(pUWOC,'Fs_dac'), pUWOC.Fs_dac = pDACSampleRate;  end
    if strcmp(pUWOC.SNR_ref,'software'), pUWOC.SNR_dB = pSNRdB; end   % same real SNR as the 'software' channel
    [rxWaveform, uwocInfo] = uwoc_vlc_chain(waveform,cInfo.SampleRate,pUWOC);

    % Signals used by the debug spectra (same orientation as the b2b branch)
    waveform_up = uwocInfo.waveform_up;
    waveform_pb = uwocInfo.waveform_pb;
    rxWaveform_pb = uwocInfo.rx_elec.';
    rxWaveform_up = uwocInfo.rxWaveform_up.';
else
    %% B2B Pass band simulation
    % Upsample the baseband OFDM symbol to the DAC Sample Rate
    [P1bb, Q1bb] = rat(pDACSampleRate/cInfo.SampleRate);
    waveform_up = resample(waveform,P1bb,Q1bb);
    
    % Apply DUC - Digital Upconverter
if hasModDemod
        waveform_pb = modulate(real(waveform_up),...
            pVLCCenterFrequency,...
            pDACSampleRate,'qam',imag(waveform_up));
else
    waveform_pb = custom_modulate(real(waveform_up),...
        pVLCCenterFrequency,...
        pDACSampleRate,'qam',imag(waveform_up));
end
    
    % B2B Test
    rxWaveform_pb = waveform_pb.';
    
    % Demodulate Back to the Baseband
if hasModDemod
        [rxReal,rxImag] = demod(rxWaveform_pb,...
            pVLCCenterFrequency,pDACSampleRate,'qam');
else
    [rxReal,rxImag] = custom_demod(rxWaveform_pb,...
        pVLCCenterFrequency,pDACSampleRate,'qam');
end
    rxWaveform_up = rxReal + 1i.*rxImag;
    
    % Downsample
    rxWaveform = resample(rxWaveform_up,Q1bb,P1bb);
    
    % Convert to column vector
    rxWaveform = rxWaveform.';
end

% Reception Normalization
rxWaveform = rxWaveform/max(abs(rxWaveform));
rxWaveform = rxWaveform - mean(rxWaveform);

%% Reception

% Get SSB from Resource Block
refGrid = DLgrid(:,1:4);

% Timing estimation. This is the timing offset to the OFDM symbol prior to
% the detected SSB due to the content of the reference grid
if has5GToolbox
    [timingOffset,mag] = nrTimingEstimate(cCarrier,rxWaveform,refGrid);
else
    [timingOffset,mag] = custom_nrTimingEstimate(cCarrier,rxWaveform,refGrid);
end

% Exclusive for SDR reception
if timingOffset > length(rxWaveform)/2
    rxWaveform_half = rxWaveform(1:end/2);
    
    % Time estimation of the first half
if has5GToolbox
        [timingOffset,mag] = nrTimingEstimate(cCarrier,rxWaveform_half,refGrid);
else
    [timingOffset,mag] = custom_nrTimingEstimate(cCarrier,rxWaveform_half,refGrid);
end
end

% Time offset Correction
rxWaveform_TC = rxWaveform(timingOffset+1:timingOffset+length(waveform));

fprintf('Timing Offset Sample: %d\n',timingOffset);

% Coarse Frequecy Offset Correction 
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
        
        % TX and RX use the same ADALM-Pluto therefore uses the same LO
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

if has5GToolbox
    % Detect PSS NID2
    NID2 = detectPSS(rxGrid,pSizeGrid);

    % Extract the received SSS symbols from the SS/PBCH block
    pbCHoffset = round((pSizeGrid*12 - 240)/2);
    startSSS = pbCHoffset + 57;
    sssRx = rxGrid(startSSS:startSSS+126,4);

    % Correlate SSS Symbol and detect Cell
    [NID1,rxCellid] = detecSSS(sssRx,NID2,true);

    % Decode PSDCH symbols
    % Get only PDSCH after SS Burst
    rxGridPDSCH = rxGrid(:,15:end);
    pdschDmrsIndices = nrPDSCHDMRSIndices(cCarrier,pdsch);
    pdschDmrsSymbols = nrPDSCHDMRS(cCarrier,pdsch);

    % Channel Estimation
    [hest,nVar,pdschHestInfo] = nrChannelEstimate(rxGridPDSCH,pdschDmrsIndices,pdschDmrsSymbols);

    % Get PDSCH Indices and symbols
    [pdschIndices,pdschIndicesInfo] = nrPDSCHIndices(cCarrier,pdsch);
    [pdschRxSym,pdschHest] = nrExtractResources(pdschIndices,rxGridPDSCH,hest);

    % Apply Equalizer
    pdschEqSym = nrEqualizeMMSE(pdschRxSym,pdschHest,nVar);

    % Recover bits
    % Create DLSCH decoder object
    decodeDLSCH = nrDLSCHDecoder;
    decodeDLSCH.MultipleHARQProcesses = false;
    decodeDLSCH.TargetCodeRate = pCodeRate;
    decodeDLSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
    decodeDLSCH.MaximumLDPCIterationCount = 6;

    % Decode the equalized PDSCH Symbols and recover the codeword
    [dlschLLRs,rxSymbols] = nrPDSCHDecode(cCarrier,pdsch,pdschEqSym,nVar);
    decodeDLSCH.TransportBlockLength = trBlkSizes;
    [decbits,blkerr] = decodeDLSCH(dlschLLRs,pdsch.Modulation,pdsch.NumLayers,0);

    %% Results
    % Bit Error Rate Calculation
    [num_erros,ber] = biterr(trBlk,decbits);
    output = sprintf('BER = %.4f, Errors = %d', ber, num_erros);
    disp(output);

    % Calculate EVM
    [EVM_dB, EVM_porc] = evm(pModulationLevel,pdschSymbols,pdschEqSym);
    fprintf('EVM Percentual = %.2f',EVM_porc);
else
    % Execucao via funcoes custom (PSS/SSS, canal por DMRS, MMSE, BER pre-FEC, EVM)
    [rxCellid, pdschRxSym, pdschEqSym, hest, decbits, ber, EVM_porc] = ...
        custom_downlink_rx(rxGrid, pSizeGrid, pModulationLevel, trBlk, pNCellID, txInfo);
end


%% Plot Figures
%% Show TX/RX spectra
figure();
subplot(2,1,1);
if    ~strcmp(pChannel,'adalm2000')
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
if strcmp(pChannel,'adalm2000')
    %xlim([0,pVLCSampleRate/2]);
end

subplot(2,1,2);
if   ~strcmp(pChannel,'adalm2000')
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
if strcmp(pChannel,'adalm2000')
    %xlim([0,pVLCSampleRate/2]);
end

if isOctave
    drawnow;
    quadro = getframe(gcf);
    nome_arquivo = sprintf('Figura2_%s_%s.png', pChannel, pDigitalModulation);
    imwrite(quadro.cdata, nome_arquivo);
    fprintf('Figura salva: %s\n', nome_arquivo);
end   

%% Resource Grid Plot
figure();
imagesc(abs(DLgrid));
%caxis([0 4]);
axis xy;
xlabel('OFDM symbol');
ylabel('Subcarrier');
title('5G NR Resource Grid - SSB + PDCCH + PDSCH')

if isOctave
    drawnow;
    quadro = getframe(gcf);
    nome_arquivo = sprintf('Figura3_%s_%s.png', pChannel, pDigitalModulation);
    imwrite(quadro.cdata, nome_arquivo);
    fprintf('Figura salva: %s\n', nome_arquivo);
end   

%% Time Domain Plot
figure()
plot(real(waveform))
hold on
plot(real(rxWaveform_TCFC))
plot(real(rxWaveform_TC))
legend('TX', 'RX_TCFC', 'RX_TC')

if isOctave
    drawnow;
    quadro = getframe(gcf);
    nome_arquivo = sprintf('Figura4_%s_%s.png', pChannel, pDigitalModulation);
    imwrite(quadro.cdata, nome_arquivo);
    fprintf('Figura salva: %s\n', nome_arquivo);
end   

%% Constelation Diagram Plot
if has5GToolbox
    rxGridTC = nrOFDMDemodulate(cCarrier,rxWaveform_TC);
    rxGridPDSCH_TC = rxGridTC(:,15:end);
    [pdschRxSymTC,~] = nrExtractResources(pdschIndices,rxGridPDSCH_TC);
else
    rxGridTC = custom_nrOFDMDemodulate(cCarrier,rxWaveform_TC);
    rxGridPDSCH_TC = rxGridTC(:,15:end);
    pdschRxSymTC = rxGridPDSCH_TC(txInfo.pdschIndices);
end

figure()
if has5GToolbox
    txSymbols = pdschGrid(pdschIndices);
else
    txSymbols = pdschSymbols;
end
subplot(2,2,4);
plot(real(pdschEqSym),imag(pdschEqSym),'b.');
if has5GToolbox
    a = max(max(real(rxSymbols{1,1})));
else
    a = max(max(real(pdschEqSym)));
end
axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
title ('Const RX Time and Freq Sync + Eq')
xlabel('Real'), ylabel('Imag'), grid, hold on

subplot(2,2,1);
plot(real(txSymbols),imag(txSymbols),'r.');
title ('Const TX')
axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
xlabel('Real'), ylabel('Imag'), grid, hold on

subplot(2,2,3);
plot(real(pdschRxSym),imag(pdschRxSym),'g.');
title ('Const RX with Time and Freq Sync')
%axis ([-a-1 a+1 -a-1 a+1]);
xlabel('Real'), ylabel('Imag'), grid, hold on

subplot(2,2,2);
plot(real(pdschRxSymTC),imag(pdschRxSymTC),'k.');
title ('Const RX with Time Sync')
%axis ([-a-1 a+1 -a-1 a+1]);
xlabel('Real'), ylabel('Imag'), grid, hold on

if isOctave
    drawnow;
    quadro = getframe(gcf);
    nome_arquivo = sprintf('Figura5_%s_%s.png', pChannel, pDigitalModulation);
    imwrite(quadro.cdata, nome_arquivo);
    fprintf('Figura salva: %s\n', nome_arquivo);
end   

%% Estimated Channel
figure();
mesh(abs(hest(:,:,1,1)));
title('Channel Estimate');
xlabel('OFDM Symbol');
ylabel("Subcarrier");
zlabel("Magnitude");
if isOctave
    drawnow;
    quadro = getframe(gcf);
    nome_arquivo = sprintf('Figura6_%s_%s.png', pChannel, pDigitalModulation);
    imwrite(quadro.cdata, nome_arquivo);
    fprintf('Figura salva: %s\n', nome_arquivo);
end   

%% Debug ADALM 2000 spectra
if strcmp(pChannel,'adalm2000') || strcmp(pChannel,'b2b') || strcmp(pChannel,'uwoc-b2b')
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

    if isOctave
        drawnow;
        quadro = getframe(gcf);
        nome_arquivo = sprintf('Figura7_%s_%s.png', pChannel, pDigitalModulation);
        imwrite(quadro.cdata, nome_arquivo);
        fprintf('Figura salva: %s\n', nome_arquivo);
    end       
end

if strcmp(pChannel,'adalm2000') || strcmp(pChannel,'b2b') || strcmp(pChannel,'uwoc-b2b')
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

    if isOctave
        drawnow;
        quadro = getframe(gcf);
        nome_arquivo = sprintf('Figura8_%s_%s.png', pChannel, pDigitalModulation);
        imwrite(quadro.cdata, nome_arquivo);
        fprintf('Figura salva: %s\n', nome_arquivo);
    end       
    
end