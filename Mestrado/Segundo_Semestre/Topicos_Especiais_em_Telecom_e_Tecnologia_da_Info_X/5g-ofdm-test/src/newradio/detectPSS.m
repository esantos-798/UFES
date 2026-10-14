function outNID = detectPSS(rxGrid,SizeGrid)
% Detect PSS with cross correlation

% Input Parameters
% rxGrid - Received Resource Grid
% SizeGrid - Size of the received grid

% Output Parameters
% outNID - NID2 detect in PSS

% Create reference grid for timing estimation containing one PSS. The
refGrid = zeros([SizeGrid*12 2]);

% PSS Search
peakCorr = 0;
outNID = 0;
pbCHoffset = round((SizeGrid*12 - 240)/2);
startPSS = pbCHoffset + 57;

% Search for NID2
for NID2 = [0 1 2]
    
    refGrid(startPSS:startPSS+126,2) = nrPSS(NID2);
    corr = xcorr(rxGrid(:,2),refGrid(:,2));
    if (max(corr) > peakCorr)
        peakCorr = max(corr);
        outNID = NID2;
    end
end
end

