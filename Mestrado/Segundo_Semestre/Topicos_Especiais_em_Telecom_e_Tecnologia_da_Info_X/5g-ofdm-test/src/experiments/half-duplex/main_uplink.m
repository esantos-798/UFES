% Test 5G New Radio over ADALM Pluto/2000
% Developed by Augusto Peterle
% 14/04/2023

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
pChannel = 'adalm-pluto';          % Channel can be: 'software', 'adalm-pluto', 'adalm2000', 'b2b'
pBW = 5e6;                      % Output element Bandwidth
pLayers = 1;                    % Number of layers send over the antenna
pDigitalModulation = 'QPSK';   % Digital modulation: 'qam' or 'psk'
pModulationLevel = 4;          % Modulation M-QAM or M-PSK
pCyclicPrefix = 'Normal';       % Cyclic Prefix 'Normal' or 'Extended'
pNCellID = 102;                 % Number of Cell ID used in PBCH
pCodeRate = 490/1024;           % Code Rate

% Software Simulation parameters
pSNRdB = 10;                    % SNR for AWGN in dBrefGrid
pFrequencyOffset = 1.2e3;       % Frequency offset (Hz)
pPhaseOffset = 15;              % Phase offset (Degrees)

% ADALM PLUTO Parameters
pGain = -25;                    % Transmitter Gain in dB
pCenterFrequency = 2e9;       % Pass-band Center Frequency
%pRadioIDTx = 'usb:0';           % Transmitter Radio ID
%pRadioIDRx = 'usb:1';           % Receiver Radio ID
pRadioIDTx = "usb:0";           % Transmitter Radio ID
pRadioIDRx = "usb:0";           % Receiver Radio ID

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

%% Uplink Transmission

% Create SRS - Sound Reference Signal to Synchronize Uplink Transmission
srs = nrSRSConfig('SymbolStart',1','FrequencyStart',12);

% Create SRS Resource Grid
srsGrid = mapSRS(cCarrier,srs,1,false);

% Create PUCCH - Physycal Uplink Control Channel
pucch = nrPUCCH2Config('NSizeBWP',pSizeGrid,'NStartBWP',0,...
    'SymbolAllocation',[3,2],'PRBSet',12);
uciBits = [1;1];
pucchGrid = mapPUCCH(cCarrier,pucch,uciBits,false);

% Create Physical Uplink Shared Channel (PUSCH) Configuration Instance
pusch = nrPUSCHConfig('NumLayers',pLayers,'PRBSet',[0:1:pSizeGrid-1]...
    ,'Modulation',pDigitalModulation);

% Testing with PUSCH
pusch.DMRS.DMRSTypeAPosition = 2;      % 2 or 3
pusch.DMRS.DMRSLength = 1;             % 1 or 2
pusch.DMRS.DMRSAdditionalPosition = 1; % 0...3


% Get Available Indices of PUSCH and PUSCH Info
[puschIndices,puschInfo] = nrPUSCHIndices(cCarrier,pusch);
payloadLengthBits = puschInfo.G;

% Generate a UL-SCH object and the Transport Block Size
[encodeULSCH,trBlkSizes] = initULSCH(pCodeRate,pusch,cCarrier);

% New Random Transport Block
trBlk = randi([0 1],trBlkSizes,1);
setTransportBlock(encodeULSCH,trBlk);

% Encode the Transport Block and map into digital symbols
codedTrBlock = encodeULSCH(pusch.Modulation,1,puschInfo.G,0);
puschSymbols = nrPUSCH(cCarrier,pusch,codedTrBlock);

% Map PUSCH symbols to a resource Grid
puschGrid = mapPUSCH(cCarrier,pusch,puschSymbols,...
    puschIndices,pLayers,false);

% Map FULL Uplink grid
ULGrid = mapULGrid(srsGrid,puschGrid,pucchGrid, true);

% Uplink OFDM Modulation
[waveform,cInfo] = nrOFDMModulate(cCarrier,ULGrid,'Windowing',0);

% Normalization
waveform = waveform/(max(abs(waveform)));
%waveform = waveform - mean(waveform);

%% Transmission Channel

if strcmp(pChannel,'software')
    %% Software simulation
    % Add random offset with noise
    offset = randi([0 1e2]); 
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
    
    rxWaveform = rxWaveform/max(abs(rxWaveform));
    %rxWaveform = rxWaveform - mean(rxWaveform);
    
elseif strcmp(pChannel,'adalm2000')
    %% VLC Channel with ADALM 2000
    
    % Upsample the baseband OFDM symbol to the DAC Sample Rate
    [P1bb, Q1bb] = rat(pDACSampleRate/cInfo.SampleRate);
    waveform_up = resample(waveform,P1bb,Q1bb);
    
    % Apply DUC - Digital Upconverter - Analogical QAM
    waveform_pb = modulate(real(waveform_up),...
        pVLCCenterFrequency,...
        pDACSampleRate,'qam',imag(waveform_up));
   
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

% Get SSB from Resource Block
refGrid = ULGrid(:,1:3);

% Timing estimation. This is the timing offset to the OFDM symbol prior to
% the detected SSB due to the content of the reference grid
[timingOffset,mag] = nrTimingEstimate(cCarrier,rxWaveform,refGrid);

% Exclusive for SDR reception
if timingOffset > length(rxWaveform)/2
    rxWaveform_half = rxWaveform(1:end/2);
    
    % Time estimation of the first half
    [timingOffset,mag] = nrTimingEstimate(cCarrier,rxWaveform_half,refGrid);
end

% Time offset Correction
rxWaveform_TC = rxWaveform(timingOffset+1:timingOffset+length(waveform));

fprintf('Timing Offset Sample: %d\n',timingOffset);

% Coarse Frequecy Offset Correction 
frequencyCorrectionRange = -5e3:1e3:5e3;
[correctedWaveform,appliedFrequencyCorrection] = nrOFDM_CFO(rxWaveform_TC,...
    frequencyCorrectionRange,refGrid,cInfo.SampleRate,cCarrier);
fprintf(' Coarse Frequency Correction : %d\n',appliedFrequencyCorrection);

% Fine Frequency Offset Correction
frequencyCorrectionRange = -1000:5:1000;
[rxWaveform_TCFC,appliedFrequencyCorrection] = nrOFDM_CFO(correctedWaveform,...
    frequencyCorrectionRange,refGrid,cInfo.SampleRate,cCarrier);
fprintf(' Fine Frequency Correction : %d\n',appliedFrequencyCorrection);

% OFDM Demodulate
if strcmp(pChannel,'adalm-pluto')
    if strcmp(txPluto.RadioID,rxPluto.RadioID)
        
        % TX and RX use the same ADALM-Pluto therefore uses the same LO
        rxGrid = nrOFDMDemodulate(cCarrier,rxWaveform_TC,'SampleRate',cInfo.SampleRate);
    else
        rxGrid = nrOFDMDemodulate(cCarrier,rxWaveform_TCFC,'SampleRate',cInfo.SampleRate);
    end
else
    rxGrid = nrOFDMDemodulate(cCarrier,rxWaveform_TCFC,'SampleRate',cInfo.SampleRate);
end

% Decode PSDCH symbols
% Get only PDSCH after SS Burst
rxGridPUSCH = rxGrid(:,15:end);
puschDmrsIndices = nrPUSCHDMRSIndices(cCarrier,pusch);
puschDmrsSymbols = nrPUSCHDMRS(cCarrier,pusch);

% Channel Estimation
[hest,nVar,puschHestInfo] = nrChannelEstimate(rxGridPUSCH,puschDmrsIndices,puschDmrsSymbols);

% Get PDSCH Indices and symbols
[puschIndices,pdschIndicesInfo] = nrPUSCHIndices(cCarrier,pusch);
[puschRxSym,puschHest] = nrExtractResources(puschIndices,rxGridPUSCH,hest);

% Apply Equalizer
puschEqSym = nrEqualizeMMSE(puschRxSym,puschHest,nVar);

% Recover bits
% Create DLSCH decoder object
decodeULSCH = nrULSCHDecoder;
decodeULSCH.MultipleHARQProcesses = false;
decodeULSCH.TargetCodeRate = pCodeRate;
decodeULSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
decodeULSCH.MaximumLDPCIterationCount = 6;

% Decode the equalized PDSCH Symbols and recover the codeword
[ulschLLRs,rxSymbols] = nrPUSCHDecode(cCarrier,pusch,puschEqSym,nVar);
decodeULSCH.TransportBlockLength = trBlkSizes;
[decbits,blkerr] = decodeULSCH(ulschLLRs,pusch.Modulation,pusch.NumLayers,0);

%% Results
% Bit Error Rate Calculation
[num_erros,ber] = biterr(trBlk,decbits);
output = sprintf('BER = %.4f, Errors = %d', ber, num_erros);
disp(output);

% Calculate EVM
[EVM_dB, EVM_porc] = evm(pModulationLevel,puschSymbols,puschEqSym);
fprintf('EVM Percentual = %.2f\n',EVM_porc);


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

%% Time Domain Plot
figure()
plot(real(waveform))
hold on
plot(real(rxWaveform_TCFC))
plot(real(rxWaveform_TC))
legend('TX', 'RX_TCFC', 'RX_TC')

%% Constelation Diagram Plot
rxGridTC = nrOFDMDemodulate(cCarrier,rxWaveform_TC);
rxGridPUSCH_TC = rxGridTC(:,15:end);
[puschRxSymTC,~] = nrExtractResources(puschIndices,rxGridPUSCH_TC);

figure()
txSymbols = puschGrid(puschIndices);
subplot(2,2,4);
plot(real(puschEqSym),imag(puschEqSym),'b.');
a = max(max(real(rxSymbols)));
axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
title ('Const RX Time and Freq Sync + Eq')
xlabel('Real'), ylabel('Imag'), grid, hold on

subplot(2,2,1);
plot(real(txSymbols),imag(txSymbols),'r.');
title ('Const TX')
axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
xlabel('Real'), ylabel('Imag'), grid, hold on

subplot(2,2,3);
plot(real(puschRxSym),imag(puschRxSym),'g.');
title ('Const RX with Time and Freq Sync')
%axis ([-a-1 a+1 -a-1 a+1]);
xlabel('Real'), ylabel('Imag'), grid, hold on

subplot(2,2,2);
plot(real(puschRxSymTC),imag(puschRxSymTC),'k.');
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
