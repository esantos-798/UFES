function pucchGrid = mapPUCCH(carrier,pucch,uciBits,plotConfig)

% Map Physycal Uplink Control Channel and returns the pucch grid

% Input Parameters:
% carrier - nrCarrierConfig structure
% pucch - nrPUCCHConfig structure
% uciBits - Uplink Control Information Bits
% plotConfig - True or False

% Output Parameters:
% pucchGrid - Physical Uplink Control Channel Resource Grid

% Init a Resource Grid
pucchGrid = nrResourceGrid(carrier,1);

% Encode UCI
% The codeword should be 2x the Symbol idx size
E = 32;
codeduci = nrUCIEncode(uciBits,E);

% Scrambling and generate symbols
sym = nrPUCCH(carrier,pucch,codeduci);
Idx = nrPUCCHIndices(carrier,pucch);
pucchGrid(Idx) = 1*sym;

% Generate Demodulation Reference Signal (DM-RS)
dmrs = nrPUCCHDMRS(carrier,pucch);
dmrsIdx = nrPUCCHDMRSIndices(carrier,pucch);
pucchGrid(dmrsIdx) = 1*dmrs;

if(plotConfig)
    figure()
    imagesc([0 carrier.SymbolsPerSlot-1],[0 carrier.NSizeGrid*12-1],abs(pucchGrid(:,:,1)));
    axis xy;title("Resource Grid (First Antenna) - PUCCH Symbols");
        xlabel("OFDM Symbol");ylabel("Subcarrier")
end
end

