function [outBit,decUCI,outBlkErr,rxSymbolsEq,rxSymbolsnEq,rxWaveform_TCFC,hest,outRawBit] = ...
    decodeUplink(gNB,gNBConfig,rxWaveform,pChannel)

% Decode the downlink reception
% Input Parameters
% gNB - gNodeB instance
% gNBConfig - gNodeB Configuration instance
% rxWaveform - complex double column array with OFDM received signal
% pChannel - software or adalm channel

% Output Parameters
% outbit - decoded pusch transport block
% decUCI - decoded UplinkControl Information
% outBlkErr - error detection in received transport blocks
% rxSymbols - modulated rx symbols
% rxCellID - cell ID detected

% Create SRS - Sound Reference Signal to Synchronize Uplink Transmission
srs = nrSRSConfig('SymbolStart',1','FrequencyStart',12);

% Create SRS Resource Grid
srsGrid = mapSRS(gNB.cCarrier,srs,1,false);
refGrid = srsGrid(:,1:2);

% Timing estimation. This is the timing offset to the OFDM symbol prior to
% the detected SSB dgNB to the content of the reference grid
[timingOffset,~] = nrTimingEstimate(gNB.cCarrier,rxWaveform,refGrid);

% Exclusive for SDR reception
if timingOffset > length(rxWaveform)/2
    rxWaveform_half = rxWaveform(1:end/2);
    
    % Time estimation of the first half
    [timingOffset,~] = nrTimingEstimate(gNB.cCarrier,rxWaveform_half,refGrid);
end

% Time offset Correction
txWaveformSize = gNB.cSampleRate/100;
rxWaveform_TC = rxWaveform(timingOffset+1:timingOffset+txWaveformSize);
fprintf('Timing Offset Sample: %d\n',timingOffset);

% Coarse FreqgNBcy Offset Correction 
frequencyCorrectionRange = -8e3:1e3:8e3;
%frequencyCorrectionRange = -4e3:1e3:4e3;
[correctedWaveform,appliedFrequencyCorrection] = nrOFDM_CFO(rxWaveform_TC,...
    frequencyCorrectionRange,refGrid,gNB.cSampleRate,gNB.cCarrier);
fprintf(' Coarse Frequency Correction : %d\n',appliedFrequencyCorrection);

% Fine FreqgNBncy Offset Correction
frequencyCorrectionRange = -700:10:700;
[rxWaveform_TCFC,appliedFrequencyCorrection] = nrOFDM_CFO(correctedWaveform,...
    frequencyCorrectionRange,refGrid,gNB.cSampleRate,gNB.cCarrier);
fprintf(' Fine Frequency Correction : %d\n',appliedFrequencyCorrection);

% OFDM Demodulate
rxGrid = nrOFDMDemodulate(gNB.cCarrier,rxWaveform_TCFC,...
    'SampleRate',gNB.cSampleRate);

% Decode PUCCH Symbols on First Slot
rxGridPUCCH = rxGrid(:,1:14);

% Decode the PUCCH ACK/NACK
[decUCI,~] = demapPUCCH(gNB.cCarrier,rxGridPUCCH);


% Decode PUSCH symbols
% Get only pusch after SS Burst
rxGridPUSCH = rxGrid(:,15:end);
[outBit,outBlkErr,rxSymbolsEq,rxSymbolsnEq,hest,outRawBit] = demapPUSCH(gNB,gNBConfig,rxGridPUSCH);



end

