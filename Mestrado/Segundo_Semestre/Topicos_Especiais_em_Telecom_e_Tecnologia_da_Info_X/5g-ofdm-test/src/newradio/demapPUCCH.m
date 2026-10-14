function [uci,err] = demapPUCCH(carrier,rxPucchGrid)

% Map Downlink Control Information block to Physical Downlink Control
% Channel

% Input Parameters:
% carrier - nrCarrierConfig structure
% rxPdcchGrid - Received Physical Downlink Control Channel to be decoded

% Output parameters:
% uci - Uplink Control Information Recovered
% err - check CRC for errors in UCI

% Fixed Parameters
E = 32;         % Codeword size

% Create PUCCH - Physycal Uplink Control Channel
pucch = nrPUCCH2Config('NSizeBWP',carrier.NSizeGrid,'NStartBWP',0,...
    'SymbolAllocation',[3,2],'PRBSet',12);

% Generate Demodulation Reference Signal (DM-RS)
dmrsSym = nrPUCCHDMRS(carrier,pucch);
dmrsIdx = nrPUCCHDMRSIndices(carrier,pucch);
ind = nrPUCCHIndices(carrier,pucch);

% Channel Estimation
[hest,nVar,~] = nrChannelEstimate(rxPucchGrid,dmrsIdx,dmrsSym);

% Extract PDCCH received symbols
[rxSym,pucchHest] = nrExtractResources(ind,rxPucchGrid,hest);

% Equalization
[pucchEq,~] = nrEqualizeMMSE(rxSym,pucchHest,nVar);

% Demodulate
rxUCIScr = nrPUCCHDecode(carrier,pucch,E,pucchEq); 
rxUCIScr = double(rxUCIScr{1}<0);

% Decode
[uci,err] = nrUCIDecode(1-2*rxUCIScr,2,'ListLength',4);


end