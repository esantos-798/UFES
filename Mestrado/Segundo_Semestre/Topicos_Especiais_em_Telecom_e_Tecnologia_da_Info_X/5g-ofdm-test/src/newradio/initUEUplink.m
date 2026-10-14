function [ULGrid,ULwaveform,totalSymbols,totalBits] = initUEUplink(UE,UEConfig,uciBits,trBlk)

%INITUEUPLINK Initiate a new UE Uplink Frame

% Input Parameters
% UE - User Equipment structure
% UEConfig - User Equipment Config structure
% uciBits - Uplink Control Information bits
% trBlk - Transport Block bits

% Output Parameters
% ULGrid - Uplink Resource Grid
% ULwaveform - Uplink modulated OFDM 
% totalBits - raw bits transmitted (TrBlk with CR)

% Create PUCCH - Physycal Uplink Control Channel Grid
pucchGrid = mapPUCCH(UE.cCarrier,UE.pucch,uciBits,false);

% Get Available Indices of PUSCH and PUSCH Info
[puschIndices,puschInfo] = nrPUSCHIndices(UE.cCarrier,UE.pusch);

% Start pdschGrid as empty
puschGrid = [];

% Init total symbols variable
totalSymbols = [];
totalBits=[];
% Populate the whole PDSCH Grid
for idx=1:UE.cCarrier.SlotsPerFrame-1
    
    % Set a column vector transport block array
    setTransportBlock(UE.encodeULSCH,trBlk(:,idx));
    
    % Encode the Transport Block and map into digital symbols
    codedTrBlock = UE.encodeULSCH(UE.pusch.Modulation,...
        1,puschInfo.G,0);
    puschSymbols = nrPUSCH(UE.cCarrier,UE.pusch,codedTrBlock);

    % Map PUSCH symbols to a resource Grid
    puschGrid = [puschGrid mapPUSCH(UE.cCarrier,UE.pusch,puschSymbols,...
        puschIndices,UEConfig.pLayers,false)];
    
    % Concatenate symbols
    totalSymbols = [totalSymbols;puschSymbols];

    % Concatenate bits
    totalBits = [totalBits;codedTrBlock];
end

% Map FULL Uplink grid
ULGrid = mapULGrid(UE.srsGrid,puschGrid,pucchGrid, false);

% Uplink OFDM Modulation
[ULwaveform,~] = nrOFDMModulate(UE.cCarrier,ULGrid,'Windowing',0);

% Normalize Uplink Waveform
ULwaveform = ULwaveform/max(abs(ULwaveform));
ULwaveform = ULwaveform  - mean(ULwaveform );
end

