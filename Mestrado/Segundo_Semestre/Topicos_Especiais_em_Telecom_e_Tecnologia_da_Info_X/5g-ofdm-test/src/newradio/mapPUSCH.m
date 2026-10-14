function puschGrid = mapPUSCH(carrier,pusch,puschSymbols,puschIndices,p,plotConfig)

% Map Transport Block coded PuschSymbol to Physical Uplink Shared Channel
% Resource Grid

% Input Parameters:
% carrier - nrCarrierConfig structure
% pusch - nrPUSCHConfig structure
% puschIndices - column indice array
% p - Number of antenna layers
% plotConfig - True or False

% Output Parameters
% puschGrid - Physical Uplink Shared Channel Resource Grid

% Generate Demodulation Reference Signal (DM-RS)
dmrsSymbols = nrPUSCHDMRS(carrier,pusch);
dmrsIndices = nrPUSCHDMRSIndices(carrier,pusch);

% Init a Resource Grid
puschGrid = nrResourceGrid(carrier,1);

% Map Resource Grid Indices of the Payload Bits
[~,puschAntIndices] = nrExtractResources(puschIndices,puschGrid);
puschGrid(puschAntIndices) = puschSymbols;

% Map Resource Grid Indices of DM-RS
[~,dmrsAntIndices] = nrExtractResources(dmrsIndices(:,p),puschGrid);
puschGrid(dmrsAntIndices) = puschGrid(dmrsAntIndices) + dmrsSymbols(:,p);

% Plot Graphs
if (plotConfig)
    
    % Plot PUSCH Constellation Diagram
    figure();
    plot(puschSymbols(:),"o");hold on
    plot(dmrsSymbols(:),"xr");hold off
    title("PUSCH and PUSCH DM-RS Symbols");xlabel("In-Phase Amplitude");ylabel("Quadrature Amplitude")
    legend("PUSCH","PUSCH DM-RS")
    
    % Plot Resource Grid
    figure()
    imagesc([0 carrier.SymbolsPerSlot-1],[0 carrier.NSizeGrid*12-1],abs(puschGrid(:,:,1)));
    axis xy;title("Resource Grid (First Antenna) - PUSCH and PUSCH DM-RS");
        xlabel("OFDM Symbol");ylabel("Subcarrier")
end

end

