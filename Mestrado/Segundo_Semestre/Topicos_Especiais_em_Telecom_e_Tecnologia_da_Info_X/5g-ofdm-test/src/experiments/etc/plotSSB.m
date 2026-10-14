clear all
close all
clc

%% Initialization 
pSCS = 15;                      % Subcarrier spacing
pSizeGrid = 25;                 % Size grid - Number of resource blocks
pChannel = 'software';         % Channel can be: 'software', 'adalm-pluto', 'adalm2000', 'b2b'
pBW = 5e6;                      % Output element Bandwidth
pLayers = 1;                    % Number of layers send over the antenna
pDigitalModulation = 'QPSK';    % Digital modulation: 'qam' or 'psk'
pModulationLevel = 4;           % Modulation M-QAM or M-PSK
pCyclicPrefix = 'Normal';       % Cyclic Prefix 'Normal' or 'Extended'
pNCellID = 102;                 % Number of Cell ID used in PBCH
pCodeRate = 679/1024;           % Code Rate
ncellid = 102;
plotConfig = true;
%% Initialization Calcs

% Calculate Effective Bandwidth
cBW = pSCS*1e3*12*pSizeGrid;    

% Init 5GNR Carrier
carrier = nrCarrierConfig('SubcarrierSpacing',pSCS,'NSizeGrid',pSizeGrid);

% Get OFDM Info
cInfo = nrOFDMInfo(carrier);

% SampleRate
cSampleRate = cInfo.SampleRate;

% Init SS Block
ssblock = zeros([240 4]);

% Init PSS - Primary Synchronization Signal
pssSymbols = nrPSS(ncellid);
pssIndices = nrPSSIndices;

% Init SSS - Secondary Synchronization Signal
sssSymbols = nrSSS(ncellid);
sssIndices = nrSSSIndices;

% Init PBCH - Physical Broadcast Channel
% Codeword of length 864 bits, created by performing BCH encoding of the MIB
cw = randi([0 1],864,1);

% Scrambling and modulation for PBCH 
v = 0;
pbchSymbols = nrPBCH(cw,ncellid,v);
pbchIndices = nrPBCHIndices(ncellid);

% DM-RS Symbols
ibar_SSB = 0;
dmrsSymbols = nrPBCHDMRS(ncellid,ibar_SSB);
dmrsIndices = nrPBCHDMRSIndices(ncellid);

% Include PSS into the SS Block
ssblock(sssIndices) = 2 * sssSymbols;
ssblock(pssIndices) = 1 * pssSymbols;
ssblock(dmrsIndices) = 4 * dmrsSymbols;
ssblock(pbchIndices) = 3 * pbchSymbols;

% SSB formats
nSubcarriers= carrier.NSizeGrid*12;
s = size(ssblock);
nSubcarriersSSB= s(1);
ssbSymbolNumber = s(2);

% Create reference downlink grid
ssbGrid = nrResourceGrid(carrier);
offset = round((nSubcarriers - nSubcarriersSSB)/2) + 1;
ssbGrid(offset:nSubcarriersSSB+offset-1,2:ssbSymbolNumber+1) = ...
    ssblock(:,1:ssbSymbolNumber);

% Plot SSB Resource Grid
if plotConfig
    figure()
    imagesc(abs(ssbGrid));
    caxis([0 4]);
    axis xy;
    xlabel('OFDM symbol');
    ylabel('Subcarrier');
    str = sprintf('SS/PBCH block containing PSS, SSS, PBCH and PBCH DM-RS');
    %title(str);
end