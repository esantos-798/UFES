function [NID2,outTimeOffset] = detectStart(carrier,rxWaveform)
% Detect the Frame Start with cross-correlation with Zadoff-Chu sequence
% in Primary Synchronization Signal
% There are NID2 = [0,1,2] possibilities
% The function generate these three PSS and calculate the best max
% correlation

% Input Parameters
% carrier - 5G NR carrier structure
% rxWaveform - complex double column vector

% Output parameters
% NID2 - cell ID in PSS
% outTimeOffset - Time offset of the Frame start

% Initialize max elements
maxMag = 0;
outTimeOffset = 0;

% Search for NID2 refgrids
for i=0:2
    refGrid = initSSB(i,carrier,false);
    refGrid = refGrid(:,1:2);
    [timingOffset,mag] = nrTimingEstimate(carrier,rxWaveform,refGrid);

    % Verify maximum cross correlation magnitude and store timeoffset
    if max(abs(mag)) > maxMag
        maxMag = max(abs(mag));
        NID2 = i;
        outTimeOffset = timingOffset;
    end
end


end

