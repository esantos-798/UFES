function grid = mapDLGrid(ssbGrid,pdschGrid,pdcchGrid,carrier, plotConfig)

%MAPDLGRID Concatenate both SSB Grid, PDCCH and PDSCH Grid to create a
%Downlink Grid

% Input Parameters
% ssbGrid - Synchronization Signal Block Resource Grid
% pdschGrid - Physical Downlink Shared Channel grid
% pdcchGrid - Physical Downlink Control Channel grid

% Output Paramters
% grid - Concatenated Downlink Resource Grid

% Start with SSB grid
grid = ssbGrid;

% Concatenate PDCCH Grid
grid(:,7) = pdcchGrid(:,1);

% Concatenate SSB and PDSCH Grids
grid = [grid pdschGrid];
end

