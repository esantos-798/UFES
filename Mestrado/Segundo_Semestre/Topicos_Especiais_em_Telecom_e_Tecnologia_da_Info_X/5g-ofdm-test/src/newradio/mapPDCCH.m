function pdcchGrid = mapPDCCH(carrier,pdcch,dciBits,plotConfig)

% Map Downlink Control Information block to Physical Downlink Control
% Channel

% Input Parameters:
% carrier - nrCarrierConfig structure
% pdcch - nrPDCCHConfig structure
% pdcchBits - DCI coded bits
% pdcchIndices - column indice array
% plot - True or False

% Output parameters:
% pdcchGrid - Physical Downlink Control Channel Resource Grid

% Init a Resource Grid
pdcchGrid = nrResourceGrid(carrier,1);

% Generate Demodulation Reference Signal (DM-RS)
[ind,dmrsSym,dmrsInd] = nrPDCCHResources(carrier,pdcch);

nID = 23;           % pdcch-DMRS-ScramblingID
rnti = 100;         % C-RNTI for PDCCH in a UE-specific search space
E = 432;            % Number of bits for PDCCH resources

% Encode DCI codeword
dciCW = nrDCIEncode(dciBits,rnti,E);

% Create PDCCH Symbols
pdcchSym = nrPDCCH(dciCW,nID,rnti);

% Set resource grid with DMRS Symbols
pdcchGrid(dmrsInd) = dmrsSym;

% Set resource grid with PDCCH Symbols
pdcchGrid(ind) = pdcchSym;

% Plot Resource Grid
if(plotConfig)
    figure()
    imagesc([0 carrier.SymbolsPerSlot-1],[0 carrier.NSizeGrid*12-1],abs(pdcchGrid(:,:,1)));
    axis xy;title("Resource Grid (First Antenna) - PDCCH and PDCCH DM-RS");
        xlabel("OFDM Symbol");ylabel("Subcarrier")
end
end
