function [DLGrid,DLwaveform,totalSymbols,totalBits] = initGNBDownlink(gNB,gNBConfig,dciBits,trBlk)

%INITGNBDOWNLINK Initate a new downlink frame
% Input Parameters
% gNB - gNodeB structure
% gNBConfig - gNodeB Config structure
% dciBits - Downlink Control Information bits
% trBlk - Transport Block bits

% Output Parameters
% DLGrid - Downlink Resource Grid
% DLwaveform - Downlink OFDM modulated waveform
% totalBits - raw bits transmitted (TrBlk with CR)

% Create PDCCH Grid
pdcchGrid = mapPDCCH(gNB.cCarrier,gNB.pdcch,dciBits,false);

% Get Available Indices of PDSCH and PDSCH Info
[pdschIndices,pdschInfo] = nrPDSCHIndices(gNB.cCarrier,gNB.pdsch);

% Start pdschGrid as empty
pdschGrid = [];
totalSymbols = [];
totalBits = [];
% Populate the whole PDSCH Grid
for idx=1:gNB.cCarrier.SlotsPerFrame-1
    
    % Set a column vector transport block array
    setTransportBlock(gNB.encodeDLSCH,trBlk(:,idx));
    
    % Encode the Transport Block and map into digital symbols
    codedTrBlock = gNB.encodeDLSCH(gNB.pdsch.Modulation,...
        1,pdschInfo.G,0);
    pdschSymbols = nrPDSCH(gNB.cCarrier,gNB.pdsch,codedTrBlock);

    % Map PDSCH symbols to a resource Grid
    pdschGrid = [pdschGrid mapPDSCH(gNB.cCarrier,gNB.pdsch,pdschSymbols,...
        pdschIndices,gNBConfig.pLayers,false)];
    
    % Concatenate symbols
    totalSymbols = [totalSymbols;pdschSymbols];

    % Concatenate bits
    auxSymbols = nrPDSCHDecode(gNB.cCarrier,gNB.pdsch,pdschSymbols);
    totalBits = [totalBits codedTrBlock];
end

% Concatenate DL Resource Grid (SSB Grid + PDCCH + PDSCH Grid)
DLGrid = mapDLGrid(gNB.ssbBlockGrid,pdschGrid,pdcchGrid,...
    gNB.cCarrier, false);

% Create OFDM Symbol from Resource Grid
[DLwaveform,~] = nrOFDMModulate(gNB.cCarrier,DLGrid,'Windowing',0);

% Normalize Downlink Waveform
DLwaveform = DLwaveform/max(abs(DLwaveform));
DLwaveform = DLwaveform - mean(DLwaveform);
end
