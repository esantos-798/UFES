function pdschGrid = mapPDSCH(carrier,pdsch,pdschSymbols,pdschIndices,p,plotConfig)

% Map the transport block to Physical Downlink Shared Channel

% Input Parameters:
% carrier - nrCarrierConfig structure
% pdsch - nrPDSCHConfig structure
% pdschBits - random bit array column
% pdschIndices - column indice array
% p - Number of antenna layers
% plotConfig - True or False

% Output Parameters
% pdschGrid - Physical Downlink Shared Channel Resource Grid

% Generate Demodulation Reference Signal (DM-RS)
dmrsSymbols = nrPDSCHDMRS(carrier,pdsch);
dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);

% Init a Resource Grid
pdschGrid = nrResourceGrid(carrier,1);

% Map Resource Grid Indices of the Payload Bits
[~,pdschAntIndices] = nrExtractResources(pdschIndices,pdschGrid);
pdschGrid(pdschAntIndices) = pdschSymbols;

% Map Resource Grid Indices of DM-RS
% [~,dmrsAntIndices] = nrExtractResources(dmrsIndices(:,p),pdschGrid);
% pdschGrid(dmrsAntIndices) = pdschGrid(dmrsAntIndices) + dmrsSymbols(:,p);
pdschGrid(dmrsIndices) = dmrsSymbols;

% Plot Configuration
if (plotConfig)
    
    % Plot PDSCH Constellation Diagram
    figure();
    plot(pdschSymbols(:),"o");hold on
    plot(dmrsSymbols(:),"xr");hold off
    title("PDSCH and PDSCH DM-RS Symbols");xlabel("In-Phase Amplitude");ylabel("Quadrature Amplitude")
    legend("PDSCH","PDSCH DM-RS")
    
    % Plot PDSCH Resource Grid
    figure()
    imagesc([0 carrier.SymbolsPerSlot-1],[0 carrier.NSizeGrid*12-1],abs(pdschGrid(:,:,1)));
    axis xy;title("Resource Grid (First Antenna) - PDSCH and PDSCH DM-RS");
        xlabel("OFDM Symbol");ylabel("Subcarrier")
end

end

