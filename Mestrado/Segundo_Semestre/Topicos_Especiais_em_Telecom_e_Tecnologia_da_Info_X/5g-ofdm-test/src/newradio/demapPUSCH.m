function [outBit,outBlkErr,rxSymbolsEq,rxSymbolsnEq,hest,outRawBit] = demapPUSCH(gNB,gNBConfig,rxGridPUSCH)
% Demap received PUSCH resource grid
% This function converts the received resource grid to bits,symbols and blk
% errors

%Input Parameters
% gNB - gNodeB structure
% gNBConfig - gNodeB Configuration strucure
% rxGridPUSCH - Received PUSCH grid not equalized

% Output Parameters
% outBit - bits recovered in reception
% outBlkerr - check the received bit erros with CRC
% rxSymbols - received modulated symbols

% Create an internal pusch to decode the payload
pusch = nrPUSCHConfig('NumLayers',gNBConfig.pLayers,'PRBSet',...
    [0:1:gNBConfig.pSizeGrid-1],...
    'Modulation',gNBConfig.pDigitalModulation);

% Additional DM-RS to improve channel estimation
pusch.DMRS.DMRSTypeAPosition = 2;      % 2 or 3
pusch.DMRS.DMRSLength = 1;             % 1 or 2
pusch.DMRS.DMRSAdditionalPosition = 1; % 0...3

% Get pusch DM-RS Symbols
puschDmrsIndices = nrPUSCHDMRSIndices(gNB.cCarrier,pusch);
puschDmrsSymbols = nrPUSCHDMRS(gNB.cCarrier,pusch);

% Create an output bit and error block
outBit = [];
outBlkErr = [];
rxSymbolsEq = [];
rxSymbolsnEq = [];
outRawBit = [];
% Demodulate All Received pusch Symbols
for idx=1:gNB.cCarrier.SymbolsPerSlot:(gNB.cCarrier.SlotsPerFrame-1)*gNB.cCarrier.SymbolsPerSlot
    
    % Reset Soft buffer to prevent noise in decoder
    resetSoftBuffer(gNB.decodeULSCH);

    % Split the pusch received grid inito groups
    currentGrid = rxGridPUSCH(:,idx:idx+gNB.cCarrier.SymbolsPerSlot-1);
    
    % Channel Estimation
    [hest,nVar,~] = nrChannelEstimate(currentGrid,puschDmrsIndices,puschDmrsSymbols);

    % Get pusch Indices and symbols
    [puschIndices,~] = nrPUSCHIndices(gNB.cCarrier,pusch);
    
    % Extract Received pusch Symbols
    [puschRxSym,puschHest] = nrExtractResources(puschIndices,...
        currentGrid,hest);
    
    % Apply Equalizer
    puschEqSym = nrEqualizeMMSE(puschRxSym,puschHest,nVar);

    % Decode the equalized pusch Symbols and recover the codeword
    [ulschLLRs,rxSymbols] = nrPUSCHDecode(gNB.cCarrier,pusch,puschEqSym,nVar);
    gNB.decodeULSCH.TransportBlockLength = gNB.trBlkSizes;
    [decbits,blkerr] = gNB.decodeULSCH(ulschLLRs,pusch.Modulation,pusch.NumLayers,0);
    
    % Concat output arrays
    outBit = [outBit decbits];
    outBlkErr = [outBlkErr;blkerr];
    rxSymbolsEq = [rxSymbolsEq;rxSymbols];
    rxSymbolsnEq = [rxSymbolsnEq;puschRxSym];
    outRawBit = [outRawBit;ulschLLRs < 0];
end
end

