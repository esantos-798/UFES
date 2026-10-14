clear all;
close all;
clc;

% Create SSB Grid
ssblock = zeros([240 4])

% Define Cell ID
ncellid = 102;

% PSS
pssSymbols = nrPSS(ncellid);
pssIndices = nrPSSIndices;
ssblock(pssIndices) = 1 * pssSymbols;

%SSS
sssSymbols = nrSSS(ncellid);
sssIndices = nrSSSIndices;
ssblock(sssIndices) = 2 * sssSymbols;

% PBCH
cw = randi([0 1],864,1);
v = 0;
pbchSymbols = nrPBCH(cw,ncellid,v);
pbchIndices = nrPBCHIndices(ncellid);
ssblock(pbchIndices) = 3 * pbchSymbols;

% PBCH DMRS
ibar_SSB = 0;
dmrsSymbols = nrPBCHDMRS(ncellid,ibar_SSB);
dmrsIndices = nrPBCHDMRSIndices(ncellid);
ssblock(dmrsIndices) = 4 * dmrsSymbols;

% SS BURST
nSubframes = 5;
symbolsPerSlot = 14;

% Vou variar aqui
mu = [0,1,2,3];

figure();
for j=1:length(mu)
    nSymbols = symbolsPerSlot * 2^mu(j) * nSubframes;
    ssburst = zeros([240 nSymbols]);
    
    % SS Block Pattern
    n = [0, 1];
    firstSymbolIndex = [4; 8; 16; 20] + 28*n;
    firstSymbolIndex = firstSymbolIndex(:).'
    
    ssblock = zeros([240 4]);
    ssblock(pssIndices) = pssSymbols;
    ssblock(sssIndices) = 2 * sssSymbols;
    
    
    for ssbIndex = 1:length(firstSymbolIndex)
        
        i_SSB = mod(ssbIndex - 1,8);
        ibar_SSB = i_SSB;
        v = i_SSB;
        
        pbchSymbols = nrPBCH(cw,ncellid,v);
        ssblock(pbchIndices) = 3 * pbchSymbols;
        
        dmrsSymbols = nrPBCHDMRS(ncellid,ibar_SSB);
        ssblock(dmrsIndices) = 4 * dmrsSymbols;
        
        ssburst(:,firstSymbolIndex(ssbIndex) + (0:3)) = ssblock;
        
    end
    subplot(4,1,j);
    imagesc(abs(ssburst));
    clim([0 4]);
    axis xy;
    xlabel('OFDM symbol');
    ylabel('Subcarrier');

end
