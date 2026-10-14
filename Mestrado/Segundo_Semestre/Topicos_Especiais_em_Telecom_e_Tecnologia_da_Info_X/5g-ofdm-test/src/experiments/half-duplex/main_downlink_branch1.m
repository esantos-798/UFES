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

%% Initialization 
pSCS = 15;                      % Subcarrier spacing
pSizeGrid = 25;                 % Size grid - Number of resource blocks
pChannel = 'software';         % Channel can be: 'software', 'adalm-pluto', 'adalm2000', 'b2b'
pBW = 5e6;                      % Output element Bandwidth
pLayers = 1;                    % Number of layers send over the antenna
pDigitalModulation = 'QPSK';    % Digital modulation: 'qam' or 'psk'
pModulationLevel = 4;           % Modulation M-QAM or M-PSK
pCyclicPrefix = 'Normal';       % Cyclic Prefix 'Normal' or 'Extended'
pNCellID = 102;                 % Number of Cell ID used in PBCH
pCodeRate = 679/1024;           % Code Rate

% Software Simulation parameters
pSNRdB = 8;                   % SNR for AWGN in dBrefGrid
pFrequencyOffset = 1.2e3;       % Frequency offset (Hz)
pPhaseOffset = 15;              % Phase offset (Degrees)

% ADALM PLUTO Parameters
pGain =-20;                                  % Transmitter Gain in dB
pCenterFrequency = 2e9;                      % Pass-band Center Frequency
pRadioIDTx = "ip:192.168.70.41";             % Transmitter Radio IP
pRadioIDRx = "ip:192.168.70.61";             % Receiver Radio IP

% ADALM 2000 Parameters
pVLCCenterFrequency = 5e6;      % VLC Center Frequency
pDACSampleRate = 75e6;          % DAC Available Sample Rate
pADCSampleRate = 100e6;         % ADC Available Sample Rate
pVLCUseCustomFilter = false;    % Add custom filter to VLC transmission

%% Initialization Calcs

% Calculate Effective Bandwidth
cBW = pSCS*1e3*12*pSizeGrid;    

% Init 5GNR Carrier
cCarrier = nrCarrierConfig('SubcarrierSpacing',pSCS,'NSizeGrid',pSizeGrid);

% Get OFDM Info
cInfo = nrOFDMInfo(cCarrier);

% SampleRate
cSampleRate = cInfo.SampleRate;

%% Downlink Transmission

% Create Synchronization Signal Block (SSB) (PSS + PBCH + SSS)
ssbBlockGrid = initSSB(pNCellID,cCarrier,false);

% Create Physical Downlink Shared Channel Configuration Instance
pdsch = nrPDSCHConfig('NumLayers',pLayers,'PRBSet',[0:1:pSizeGrid-1]...
    ,'Modulation',pDigitalModulation);

% Additional DM-RS to improve channel estimation
pdsch.DMRS.DMRSTypeAPosition = 2;      % 2 or 3
pdsch.DMRS.DMRSLength = 1;             % 1 or 2
pdsch.DMRS.DMRSAdditionalPosition = 1; % 0...3

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
pdcchGrid = mapPDCCH(cCarrier,pdcch,dciBits,true);

% Concatenate DL Resource Grid (SSB Grid + PDSCH Grid)
DLgrid = mapDLGrid(ssbBlockGrid,pdschGrid,pdcchGrid,...
    cCarrier, false);

% OFDM 
[waveform,cInfo] = nrOFDMModulate(cCarrier,DLgrid,'Windowing',0);

% Normalization
waveform = waveform/(max(abs(waveform)));
waveform = waveform - mean(waveform);


%% Transmission Channel

if strcmp(pChannel,'software')
    %% Software simulation
    % Add random offset with noise
    offset = randi([1e3 1e4]); 
    rxWaveform = [zeros(offset,1);waveform];
    
    % Add white Gaussian noise to the waveform. Note that the SNR only
    rxWaveform = awgn(rxWaveform,pSNRdB,-10*log10(double(cInfo.Nfft)));
    
    % Introduce phase and frequency offsets
    PFO = comm.PhaseFrequencyOffset( ...
    'PhaseOffset',     pPhaseOffset, ...
    'FrequencyOffset', pFrequencyOffset,...
    'SampleRate',      cInfo.SampleRate);
    
    % Insert Phase and Frequency Offset
    rxWaveform = PFO(rxWaveform);
    
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
    waveform_pb = modulate(real(waveform_up),...
        pVLCCenterFrequency,...
        pDACSampleRate,'qam',imag(waveform_up));

    %waveform_pb = waveform_pb;
   
    % Configure ADALM2000 Driver
    configM2k;
    
    % Write in ADALM 2000 in channel 1
    aout.pushInterleaved(waveform_pb,1);
    rxWaveform_pb = ain.getSamplesInterleaved_matlab(5*length(waveform_pb));
    rxWaveform_pb = rxWaveform_pb.double;
    
    % Normalize and remove DC of the received signal
    rxWaveform_pb = rxWaveform_pb/max(abs(rxWaveform_pb));
    rxWaveform_pb = rxWaveform_pb - mean(rxWaveform_pb);
    
    % Resample the received signal to match ADC to DAC Sampling Rate
    [P1, Q1] = rat(pADCSampleRate/(pDACSampleRate/2));
    rxWaveform_pb = resample(rxWaveform_pb,Q1,P1);
    
    % Clear context
    clib.libm2k.libm2k.context.contextCloseAll();
    
    % Demodulate Back to the Baseband
    [rxReal,rxImag] = demod(rxWaveform_pb,...
        pVLCCenterFrequency,pDACSampleRate,'qam');
    rxWaveform_up = rxReal + 1i.*rxImag;
    
    % Downsample
    rxWaveform = resample(rxWaveform_up,Q1bb,P1bb);
    
    % Convert to column vector
    rxWaveform = rxWaveform.';
else
    %% B2B Pass band simulation
    % Upsample the baseband OFDM symbol to the DAC Sample Rate
    [P1bb, Q1bb] = rat(pDACSampleRate/cInfo.SampleRate);
    waveform_up = resample(waveform,P1bb,Q1bb);
    
    % Apply DUC - Digital Upconverter
    waveform_pb = modulate(real(waveform_up),...
        pVLCCenterFrequency,...
        pDACSampleRate,'qam',imag(waveform_up));
    
    % B2B Test
    rxWaveform_pb = waveform_pb.';
    
    % Demodulate Back to the Baseband
    [rxReal,rxImag] = demod(rxWaveform_pb,...
        pVLCCenterFrequency,pDACSampleRate,'qam');
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

% Timing estimation. This is the timing offset to the OFDM symbol prior to
% the detected SSB due to the content of the reference grid
rola=0;
[NID2,timingOffset] = detectStart(cCarrier,rxWaveform);

% Exclusive for SDR reception
if timingOffset > length(rxWaveform)/2
    rxWaveform_half = rxWaveform(1:end/2);
    
    % Time estimation of the first half
    [NID2,timingOffset] = detectStart(cCarrier,rxWaveform_half);
end

% Time offset Correction
rxWaveform_TC = rxWaveform(timingOffset+1:timingOffset+length(waveform));

fprintf('Timing Offset Sample: %d\n',timingOffset);

% Get Refgrid to detect CFO
refGrid = initSSB(NID2,cCarrier,false);
refGrid = refGrid(:,1:2);

% Coarse Frequecy Offset Correction 
frequencyCorrectionRange = -20e3:1e3:20e3;
[correctedWaveform,appliedFrequencyCorrection] = nrOFDM_CFO(rxWaveform_TC,...
    frequencyCorrectionRange,refGrid,cInfo.SampleRate,cCarrier);
fprintf(' Coarse Frequency Correction : %d\n',appliedFrequencyCorrection);

% Fine Frequency Offset Correction
frequencyCorrectionRange = -1000:5:1000;
[rxWaveform_TCFC,appliedFrequencyCorrection] = nrOFDM_CFO(correctedWaveform,...
    frequencyCorrectionRange,refGrid,cInfo.SampleRate,cCarrier);
fprintf(' Fine Frequency Correction : %d\n',appliedFrequencyCorrection);

% OFDM Demodulate
rxGrid = nrOFDMDemodulate(cCarrier,rxWaveform_TCFC,'SampleRate',cInfo.SampleRate);

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

%% Resource Grid Plot
figure();
imagesc(abs(DLgrid));
%caxis([0 4]);
axis xy;
xlabel('OFDM symbol');
ylabel('Subcarrier');
title('5G NR Resource Grid - SSB + PDCCH + PDSCH')

%% Time Domain Plot
figure()
plot(real(waveform))
hold on
plot(real(rxWaveform_TCFC))
plot(real(rxWaveform_TC))
legend('TX', 'RX_TCFC', 'RX_TC')

%% Constelation Diagram Plot
rxGridTC = nrOFDMDemodulate(cCarrier,rxWaveform_TC);
rxGridPDSCH_TC = rxGridTC(:,15:end);
[pdschRxSymTC,~] = nrExtractResources(pdschIndices,rxGridPDSCH_TC);

figure()
txSymbols = pdschGrid(pdschIndices);
subplot(2,2,4);
plot(real(rxSymbols{1,1}),imag(rxSymbols{1,1}),'b.');
a = max(max(real(rxSymbols{1,1})));
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
    plot(f,fftshift(espectroAbs));
    grid on;
    title('5G NR OFDM Baseband RX Spectrum - Banda passante');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (Hz)');
    ylim([-200,-100]);
    
    subplot(3,1,2);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(rxWaveform_up,1/(pDACSampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f,fftshift(espectroAbs));
    grid on;
    title('5G NR OFDM Baseband RX Spectrum - Upsampled');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (Hz)');
    ylim([-200,-100]);
    
    subplot(3,1,3);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(rxWaveform.',1/(cInfo.SampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f,fftshift(espectroAbs));
    grid on;
    title('5G NR OFDM Baseband RX Spectrum - Baseband');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (Hz)');
    ylim([-200,-100]);
    
end

if strcmp(pChannel,'adalm2000') || strcmp(pChannel,'b2b')
    figure();
    subplot(3,1,3);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(waveform_pb.',1/(pDACSampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f,fftshift(espectroAbs));
    grid on;
    title('5G NR OFDM Baseband TX Spectrum - Banda passante');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (Hz)');
    ylim([-200,-100]);
    
    subplot(3,1,2);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(waveform_up.',1/(pDACSampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f,fftshift(espectroAbs));
    grid on;
    title('5G NR OFDM Baseband TX Spectrum - Upsampled');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (Hz)');
    ylim([-200,-100]);
    
    subplot(3,1,1);
    [Sinal_ff,sinal_tf,f,df] = FFT_pot2(waveform.',1/(cInfo.SampleRate));
    espectroAbs = mag2db(abs(Sinal_ff));
    plot(f,fftshift(espectroAbs));
    grid on;
    title('5G NR OFDM Baseband TX Spectrum - Baseband');
    ylabel('PSD (dB/Hz)');
    xlabel('Frequêncy (Hz)');
    ylim([-200,-100]);
    
end