function [outBit,outBlkErr,rxSymbolsEq,rxSymbolsnEq,hest,outRawBit] = ...
    demapPDSCH(UE,UEConfig,rxDCI,rxGridPDSCH)
% Demap received PDSCH resource grid
% This function converts the received resource grid to bits,symbols and blk
% errors

%Input Parameters
% UE - User Equipment structure
% UEConfig - User Equipment Configuration strucure
% rxGridPDSCH - Received PDSCH grid not equalized

% Output Parameters
% outBit - bits recovered in reception
% outBlkerr - check the received bit erros with CRC
% rxSymbols - received modulated symbols

% Recover modulation index from DCI
[Qm,R,~] = getMCSLookupTable(rxDCI.Modulation);

% Create an internal PDSCH and set the modulation
pdsch = nrPDSCHConfig('NumLayers',UEConfig.pLayers,'PRBSet',...
    [0:1:UEConfig.pSizeGrid-1],...
    'Modulation',Qm);

% Additional DM-RS to decode
pdsch.DMRS.DMRSTypeAPosition = 2;      % 2 or 3
pdsch.DMRS.DMRSLength = 1;             % 1 or 2
pdsch.DMRS.DMRSAdditionalPosition = 1; % 0...3

% Get PDSCH DM-RS Symbols
pdschDmrsIndices = nrPDSCHDMRSIndices(UE.cCarrier,pdsch);
pdschDmrsSymbols = nrPDSCHDMRS(UE.cCarrier,pdsch);

% Create DLSCH decoder object and set the coderate from DCI
decodeDLSCH = nrDLSCHDecoder;
decodeDLSCH.MultipleHARQProcesses = false;
decodeDLSCH.TargetCodeRate = R;
decodeDLSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
decodeDLSCH.MaximumLDPCIterationCount = 6;
decodeDLSCH.TransportBlockLength = UE.trBlkSizes;

% Create an output bit and error block
outBit = [];
outBlkErr = [];
rxSymbolsEq = [];
rxSymbolsnEq = [];
outRawBit = [];
% Demodulate All Received PDSCH Symbols
for idx=1:UE.cCarrier.SymbolsPerSlot:(UE.cCarrier.SlotsPerFrame-1)*UE.cCarrier.SymbolsPerSlot
    
    % Reset Soft buffer to prevent noise in decoder
    resetSoftBuffer(decodeDLSCH,0);
        
    % Split the PDSCH received grid inito groups
    currentGrid = rxGridPDSCH(:,idx:idx+UE.cCarrier.SymbolsPerSlot-1);
    
    % Channel Estimation
    [hest,nVar,~] = nrChannelEstimate(currentGrid,pdschDmrsIndices,pdschDmrsSymbols);

    % Get PDSCH Indices and symbols
    [pdschIndices,~] = nrPDSCHIndices(UE.cCarrier,pdsch);
    
    % Extract Received PDSCH Symbols
    [pdschRxSym,pdschHest] = nrExtractResources(pdschIndices,...
        currentGrid,hest);
    
    % Apply Equalizer
    [pdschEqSym,~] = nrEqualizeMMSE(pdschRxSym,pdschHest,nVar);

    % Decode the equalized PDSCH Symbols and recover the codeword
    [dlschLLRs,rxSymbols] = nrPDSCHDecode(UE.cCarrier,pdsch,pdschEqSym,nVar);
    
    % Convert signaled softbits to 0-1 sequence
    hardbits = double(dlschLLRs{1,1}<0); 

    % Decode bits
    [decbits,blkerr] = decodeDLSCH(1.0 - 2*hardbits,pdsch.Modulation,pdsch.NumLayers,0);
    
    % Concat output arrays
    outBit = [outBit decbits];
    outBlkErr = [outBlkErr;blkerr];
    rxSymbolsEq = [rxSymbolsEq;rxSymbols{1,1}];
    rxSymbolsnEq = [rxSymbolsnEq;pdschRxSym];
    outRawBit = [outRawBit hardbits];
    
end

end

