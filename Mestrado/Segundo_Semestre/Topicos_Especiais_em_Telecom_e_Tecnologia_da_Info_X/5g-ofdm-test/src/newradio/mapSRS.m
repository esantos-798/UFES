function srsGrid = mapSRS(carrier,srs,p,plotConfig)

% Map Sound Reference Signal to SRS Resource Grid

% Input Parameters:
% carrier - nrCarrierConfig structure
% srs - nrSRSConfig structure
% p - Number of antenna layers
% plot - True or False

% Output Parameters:
% srsGrid - Sound Reference Signals Resource Grid

% Init a Resource Grid
srsGrid = nrResourceGrid(carrier,p);

% Extract SRS Resource index
[ind,~] = nrSRSIndices(carrier,srs);

% Generate SRS Symbols
[sym,~] = nrSRS(carrier,srs);

% Create the SRS Resource Grid
srsGrid(ind) = 1*sym;

% Plot Config
if(plotConfig)
    
    % Plot Resource Grid
    figure()
    imagesc([0 carrier.SymbolsPerSlot-1],[0 carrier.NSizeGrid*12-1],abs(srsGrid(:,:,1)));
    axis xy;title("Resource Grid (First Antenna) - SRS Symbols");
        xlabel("OFDM Symbol");ylabel("Subcarrier")
end

end

