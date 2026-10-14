function ssbGrid = initSSB(ncellid,carrier,plotConfig)
% Create a new Synschronization Block

% Input Parameters
% ncellid - Number of Cell ID
% carrier - 5G NR carrier structure
% plotConfig - true or false

% Output Parameters
% ssbGrid - Synchronization Signal Block Resource Grid

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
ssblock(sssIndices) = 1 * sssSymbols;
ssblock(pssIndices) = 1 * pssSymbols;
ssblock(dmrsIndices) = 1 * dmrsSymbols;
ssblock(pbchIndices) = 1 * pbchSymbols;

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
    title(str);
end
end

