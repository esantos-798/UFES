function [dciCW,mask] = demapPDCCH(carrier,rxPdcchGrid)

% Map Downlink Control Information block to Physical Downlink Control
% Channel

% Input Parameters:
% carrier - nrCarrierConfig structure
% rxPdcchGrid - Received Physical Downlink Control Channel to be decoded

% Output parameters:
% dciCW - decoded Downlink Controle Information
% mask - error detection 

% Fixed Parameters
nID = 23;           % pdcch-DMRS-ScramblingID
rnti = 100;         % C-RNTI for PDCCH in a UE-specific search space
E = 432;            % Number of bits for PDCCH resources
L = 8;              % Int8 size
K=32;               % 32 bits

% Create PDCCH Physical Channel with BWP full band
pdcch = nrPDCCHConfig('NStartBWP',0,'NSizeBWP',carrier.NSizeGrid);
pdcch.AggregationLevel = 4;
pdcch.CORESET.Duration = 1;
pdcch.CORESET.FrequencyResources = [1,1,1,1];

% Generate Demodulation Reference Signal (DM-RS)
[ind,dmrsSym,dmrsInd] = nrPDCCHResources(carrier,pdcch);

% Create a 14symbols received grid
pdcchGrid = nrResourceGrid(carrier,1);
pdcchGrid = pdcchGrid + rxPdcchGrid;

% Channel Estimation
[hest,nVar,~] = nrChannelEstimate(pdcchGrid,dmrsInd,dmrsSym);

% Extract PDCCH received symbols
[rxSym,pdcchHest] = nrExtractResources(ind,pdcchGrid,hest);

% Equalization
[pdcchEq,csi] = nrEqualizeMMSE(rxSym,pdcchHest,nVar);

% Demodulate
rxCW = nrPDCCHDecode(pdcchEq,nID,rnti,nVar);

% Decode DCI
[dciCW,mask] = nrDCIDecode(1-2*rxCW,K,L,rnti);

end
