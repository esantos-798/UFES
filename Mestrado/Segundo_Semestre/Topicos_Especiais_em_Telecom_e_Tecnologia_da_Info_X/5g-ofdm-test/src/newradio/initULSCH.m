function [encodeULSCH,trBlkSizes] = initULSCH(codeRate,pusch,carrier)

%INITDLSCH Initializate the Transport Uplink Shared Channel with coding
%scrambling, CRC, coderate and HARQ

% Input Parameters
% coderate - coderate of the transport block
% pusch - Physical Uplink Shared Channel structure
% carrier - 5G NR carrier structure

% Output Parameters
% encodeULSCH - object nrULSCH used to encode the transport block
% trBlkSizes - transport block bit array length

% Create UL-SCH encoder object
encodeULSCH = nrULSCH;
encodeULSCH.MultipleHARQProcesses = false;
encodeULSCH.TargetCodeRate = codeRate;

% Calculate transport block sizes
Xoh_PUSCH = 0;
[~,puschInfo] = nrPUSCHIndices(carrier,pusch);
trBlkSizes = nrTBS(pusch.Modulation,pusch.NumLayers,numel(pusch.PRBSet),puschInfo.NREPerPRB,codeRate,Xoh_PUSCH);
end

