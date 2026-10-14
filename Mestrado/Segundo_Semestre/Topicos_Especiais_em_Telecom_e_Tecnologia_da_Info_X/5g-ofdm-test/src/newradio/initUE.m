function UE = initUE(UEConfig)

%INITUE Initilize a new User Equipment Structure with physical channels
%SRS, PUCCH, PUSCH and relevant parameters

% Input Parameters
% UEConfig - User Equipment structure with relevant configurations

% Output Parameters
% UE - User equipment structure with physical channels and configurations

% Create 5GNR Carrier
UE.cCarrier = nrCarrierConfig('SubcarrierSpacing',UEConfig.pSCS,...
    'NSizeGrid',UEConfig.pSizeGrid);

% Set OFDM Info
UE.cInfo = nrOFDMInfo(UE.cCarrier);

% Set SampleRate
UE.cSampleRate = UE.cInfo.SampleRate;

% Create SRS - Sound Reference Signal to Synchronize Uplink Transmission
srs = nrSRSConfig('SymbolStart',1','FrequencyStart',12);

% Create SRS Resource Grid
UE.srsGrid = mapSRS(UE.cCarrier,srs,1,false);

% Create PUCCH - Physycal Uplink Control Channel
UE.pucch = nrPUCCH2Config('NSizeBWP',UEConfig.pSizeGrid,'NStartBWP',0,...
    'SymbolAllocation',[3,2],'PRBSet',12,'NID',1,'RNTI',1);

% Create Physical Uplink Shared Channel (PUSCH) Configuration Instance
UE.pusch = nrPUSCHConfig('NumLayers',UEConfig.pLayers,'PRBSet',[0:1:UEConfig.pSizeGrid-1]...
    ,'Modulation',UEConfig.pDigitalModulation);

% Additional DM-RS to improve channel estimation
UE.pusch.DMRS.DMRSTypeAPosition = 2;      % 2 or 3
UE.pusch.DMRS.DMRSLength = 1;             % 1 or 2
UE.pusch.DMRS.DMRSAdditionalPosition = 1; % 0...3

% Generate a UL-SCH object and the Transport Block Size
[UE.encodeULSCH,UE.trBlkSizes] = initULSCH(UEConfig.pCodeRate,...
    UE.pusch,UE.cCarrier);

end

