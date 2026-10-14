function [outBit,decDCI,outBlkErr,rxSymbolsEq,rxSymbolsnEq,...
    rxWaveform_TCFC,rxCellid,hest,outRawBit] = decodeDownlink(UE,UEConfig,rxWaveform,pChannel)

% Decode the downlink reception
% Input Parameters
% UE - User equipment instance
% UEConfig - User equipment Configuration instance
% rxWaveform - complex double column array with OFDM received signal
% pChannel - software or adalm channel

% Output Parameters
% outbit - decoded pdsch transport block
% decDCI - decoded Downlink Control Information
% outBlkErr - error detection in received transport blocks
% rxSymbols - modulated rx symbols
% rxCellID - cell ID detected

% Timing estimation. This is the timing offset to the OFDM symbol prior to
% Detect the NID2 from cell ID and the start frame offset
[NID2,timingOffset] = detectStart(UE.cCarrier,rxWaveform);

% Exclusive for SDR reception - In SDR the symbol capture are 2xlength of
% waveform
if timingOffset > length(rxWaveform)/2
    rxWaveform_half = rxWaveform(1:end/2 - 1);
    
    % Time estimation of the first half
    [NID2,timingOffset] = detectStart(UE.cCarrier,rxWaveform_half);
end

% Time offset Correction
txWaveformSize = UE.cSampleRate/100;
rxWaveform_TC = rxWaveform(timingOffset+1:timingOffset+txWaveformSize);
fprintf('Timing Offset Sample: %d\n',timingOffset);

% Detection RefGrid - PSS column grid
refGrid = initSSB(NID2,UE.cCarrier,false);
refGrid = refGrid(:,1:2);

% Coarse Frequecy Offset Correction 
frequencyCorrectionRange = -5e3:1e3:5e3;
[correctedWaveform,appliedFrequencyCorrection] = nrOFDM_CFO(rxWaveform_TC,...
    frequencyCorrectionRange,refGrid,UE.cSampleRate,UE.cCarrier);
fprintf(' Coarse Frequency Correction : %d\n',appliedFrequencyCorrection);

% Fine Frequency Offset Correction
frequencyCorrectionRange = -400:5:400;
[rxWaveform_TCFC,appliedFrequencyCorrection] = nrOFDM_CFO(correctedWaveform,...
    frequencyCorrectionRange,refGrid,UE.cSampleRate,UE.cCarrier);
fprintf(' Fine Frequency Correction : %d\n',appliedFrequencyCorrection);

% OFDM Demodulate
rxGrid = nrOFDMDemodulate(UE.cCarrier,rxWaveform_TCFC,...
    'SampleRate',UE.cSampleRate);

% Detect Cell ID

% Extract the received SSS symbols from the SS/PBCH block
pbCHoffset = round((UEConfig.pSizeGrid*12 - 240)/2);
startSSS = pbCHoffset + 57;
sssRx = rxGrid(startSSS:startSSS+126,4);

% Correlate SSS Symbol and detect Cell
[NID1,rxCellid] = detecSSS(sssRx,NID2,false);

% Decode PDCCH Symbols
rxGridPDCCH = rxGrid(:,7);
[decDCI,dciMask] = demapPDCCH(UE.cCarrier,rxGridPDCCH);
rxDCI = demapDCI(decDCI,dciMask);
rxDCI.ResourceAssignment = 15;
rxDCI.Modulation = UEConfig.DCIModulation;

% Decode PSDCH symbols
% Get only PDSCH after SS Burst
rxGridPDSCH = rxGrid(:,double(rxDCI.ResourceAssignment):end);

[outBit,outBlkErr,rxSymbolsEq,rxSymbolsnEq,hest,outRawBit] = ...
    demapPDSCH(UE,UEConfig,rxDCI,rxGridPDSCH);
end

