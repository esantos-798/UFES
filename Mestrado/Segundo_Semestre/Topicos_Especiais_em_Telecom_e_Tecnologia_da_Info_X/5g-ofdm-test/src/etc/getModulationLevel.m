function [out] = getModulationLevel(inputString)
% Get the modulation level in real 2 power of a digital modulation string

%Input:
% inputString - string that contains the digital modulation level

% Output:
% out - Modulation level
switch (inputString)

    case 'QPSK'
        out = 4;
    case '16QAM'
        out = 16;
    case '64QAM'
        out = 64;
    case '256QAM'
        out = 256;
end

