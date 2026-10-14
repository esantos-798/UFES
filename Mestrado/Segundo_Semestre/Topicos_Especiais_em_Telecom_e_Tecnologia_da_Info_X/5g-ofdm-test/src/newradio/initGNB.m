function gNB = initGNB(gNBConfig)

% Initialize a new gNB structure with parameters and strucutres
% Input Parameters
% gNBConfig - struct with gNodeB relevant configurations

% Output Parameters
% gNB - gNodeB structure with physical channels PDSCH, PDCCH, SSB, carrier
% and configurations

% Init gNB 5GNR Carrier
gNB.cCarrier = nrCarrierConfig('SubcarrierSpacing',gNBConfig.pSCS,...
    'NSizeGrid',gNBConfig.pSizeGrid);

% Set gNB OFDM Info
gNB.cInfo = nrOFDMInfo(gNB.cCarrier);

% Set gNB SampleRate
gNB.cSampleRate = gNB.cInfo.SampleRate;

% Set number of slots
gNB.NSlots = gNB.cCarrier.SlotsPerFrame;

% Create Synchronization Signal Block (SSB) (PSS + PBCH + SSS)
gNB.ssbBlockGrid = initSSB(gNBConfig.pNCellID,gNB.cCarrier,false);

% Create Physical Downlink Shared Channel Configuration Instance
gNB.pdsch = nrPDSCHConfig('NumLayers',gNBConfig.pLayers,'PRBSet',...
    [0:1:gNBConfig.pSizeGrid-1],...
    'Modulation',gNBConfig.pDigitalModulation);

% Additional DM-RS to improve channel estimation
gNB.pdsch.DMRS.DMRSTypeAPosition = 2;      % 2 or 3
gNB.pdsch.DMRS.DMRSLength = 1;             % 1 or 2
gNB.pdsch.DMRS.DMRSAdditionalPosition = 1; % 0...3

% Generate a DL-SCH object and the Transport Block Size
[gNB.encodeDLSCH,gNB.trBlkSizes] = initDLSCH(gNBConfig.pCodeRate,...
    gNB.pdsch,gNB.cCarrier);

% Create PDCCH Physical Channel
gNB.pdcch = nrPDCCHConfig('NStartBWP',0,'NSizeBWP',gNBConfig.pSizeGrid);
gNB.pdcch.AggregationLevel = 4;
gNB.pdcch.CORESET.Duration = 1;
gNB.pdcch.CORESET.FrequencyResources = [1,1,1,1];

% Create ULSCH decoder object
gNB.decodeULSCH = nrULSCHDecoder;
gNB.decodeULSCH.MultipleHARQProcesses = false;
gNB.decodeULSCH.TargetCodeRate = gNBConfig.pCodeRate;
gNB.decodeULSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
gNB.decodeULSCH.MaximumLDPCIterationCount = 6;

end

