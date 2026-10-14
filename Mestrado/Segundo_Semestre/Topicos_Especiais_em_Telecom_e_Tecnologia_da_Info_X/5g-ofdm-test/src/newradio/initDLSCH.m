function [encodeDLSCH,trBlkSizes] = initDLSCH(codeRate,pdsch,carrier)
%INITDLSCH Initializate the Transport Downlink Shared Channel with coding
%scrambling, CRC, coderate and HARQ

% Input Parameters
% coderate - coderate of the transport block
% pdsch - Physical Downlink Shared Channel structure
% carrier - 5G NR carrier structure

% Output Parameters
% encodeDLSCH - object nrDLSCH used to encode the transport block
% trBlkSizes - transport block bit array length

% Create DL-SCH encoder object
encodeDLSCH = nrDLSCH;
encodeDLSCH.MultipleHARQProcesses = false;
encodeDLSCH.TargetCodeRate = codeRate;

% Calculate transport block sizes
Xoh_PDSCH = 0;
[~,pdschInfo] = nrPDSCHIndices(carrier,pdsch);
trBlkSizes = nrTBS(pdsch.Modulation,pdsch.NumLayers,numel(pdsch.PRBSet),pdschInfo.NREPerPRB,codeRate,Xoh_PDSCH);
end

