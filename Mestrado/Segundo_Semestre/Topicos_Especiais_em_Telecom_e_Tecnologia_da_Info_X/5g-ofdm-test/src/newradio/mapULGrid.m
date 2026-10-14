function grid = mapULGrid(srsGrid,puschGrid,pucchGrid, plotConfig)

%MAPDLGRID Concatenate both SSB Grid and PDSCH Grid to create a Downlink

% Input Parameters
% srsGid - Sound Reference Signal Resource Grid - Uplink Synchronization
% puschGrid - Physical Uplink Shared Channel Resource Grid (Payload and
% DM-RS)
% pucchGrid - Physical Uplink Control Channel Resource Grid - ACK Control
% carrier - NR carrier strucutre
% plotConfig - True or False

% SRS Grid
%   SRS Grid have a specific format 
%   The SRS occupies 12 RBs

% Output Parameters
% Concatenated Uplink Grid

% First Slot is a SRS and PUCCH
grid = srsGrid + pucchGrid;

% Concatenate PUSCH
grid = [grid puschGrid];

if plotConfig
    %% Resource Grid Plot
    figure();
    imagesc(abs(grid));
    axis xy;
    xlabel('OFDM symbol');
    ylabel('Subcarrier');
    title('5G NR Resource Grid - SRS + PUCCH + PUSCH')
end

