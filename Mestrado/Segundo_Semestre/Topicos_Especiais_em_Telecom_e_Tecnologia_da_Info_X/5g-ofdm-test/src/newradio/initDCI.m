function dciBits = initDCI(modulation,newData,rv)
%INITDCI Initializate the Downlink Control Information

% Input Paramaters
% modulation - number from 0 to 30 to specify the modulation and coderate
% newData - bool value that correspond 1 = new data, 0 = retransmission
% rv - redundancy version; rv = [0,1,2]

% Output Parameters
% dciBits - Downlink Control Information bit array

%   Detailed explanation goes here
struct.Identifier = 1;              % Always 1
struct.FrequencyResources = 4;      % Formula
struct.ResourceAssignment = 15;     % First OFDM Symbol of PDSCH Resource Grid
struct.VRBPRBMap = 0;               % 0 = Non Interlevead
struct.Modulation=modulation;       % Modulation 16 QAM coderate 490/1024
struct.NewData = newData;           % 1= New Data, 0 = Retransmission
struct.RedundancyVersion = rv;      % Redundancy Version
struct.HARQ = 0;                    % HARQ Process number
struct.DownlinkIDX = 0;             % Downlink Indice counter   
struct.TPC = 0;         
struct.PUCCHResource = 0;
struct.PDSCHHARQ = 0;

% Convert the struct to a cell
cell = struct2cell(struct);

% Bit size mask
mask = [1,4,4,1,5,1,2,4,2,2,3,3];

% Make a output bit array
dciBits=[];
for i=1:length(mask)
    bits = de2bi(cell{i},mask(i));
    dciBits = [dciBits ; bits'];
end


end

