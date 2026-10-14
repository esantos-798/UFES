clear all;
close all;
clc;

% Parameters
pSCS = 15;                      % Subcarrier spacing
pSizeGrid = 25;                 % Size grid - Number of resource blocks
pDigitalModulation = '64QAM';  % Digital modulation: 'qam' or 'psk'
pLayers = 1;                    % Number of layers send over the antenna
pCodeRate = 873/1024;           % Code Rate - 490 bits uteis em 1024 bits
pSNR = 0;                       

% Newradio Carrier instance
cCarrier = nrCarrierConfig('SubcarrierSpacing',pSCS,'NSizeGrid',pSizeGrid);

% Create Physical Downlink Shared Channel Configuration Instance
pdsch = nrPDSCHConfig('NumLayers',pLayers,'PRBSet',[0:1:pSizeGrid-1]...
    ,'Modulation',pDigitalModulation);
pdsch.DMRS.DMRSTypeAPosition = 2;      % 2 or 3
pdsch.DMRS.DMRSLength = 1;             % 1 or 2
pdsch.DMRS.DMRSAdditionalPosition = 1; % 0...3

% Create DL-SCH encoder object
encodeDLSCH = nrDLSCH;
encodeDLSCH.MultipleHARQProcesses = false;
encodeDLSCH.TargetCodeRate = pCodeRate;

% Calculate transport block sizes
Xoh_PDSCH = 0;
[~,pdschInfo] = nrPDSCHIndices(cCarrier,pdsch);
trBlkSizes = nrTBS(pdsch.Modulation,pdsch.NumLayers,numel(pdsch.PRBSet),pdschInfo.NREPerPRB,pCodeRate,Xoh_PDSCH);

% New Random Transport Block - PAYLOAD
trBlk = randi([0 1],trBlkSizes,1);
setTransportBlock(encodeDLSCH,trBlk);

% APPLY CODE RATE
codedTrBlock = encodeDLSCH(pdsch.Modulation,1,pdschInfo.G,0);

% Digital mapping PDSCH symbols
pdschSymbols = nrPDSCH(cCarrier,pdsch,codedTrBlock);

% Apply white noise
rx_pdschSymbols = awgn(pdschSymbols,pSNR);

% Receive bits
[rxbits,~] = nrPDSCHDecode(cCarrier,pdsch,rx_pdschSymbols);

% Convert signaled softbits to 0-1 sequence
hardbits = double(rxbits{1,1}<0); 

% Create DLSCH decoder object
decodeDLSCH = nrDLSCHDecoder;
decodeDLSCH.MultipleHARQProcesses = false;
decodeDLSCH.TargetCodeRate = pCodeRate;
decodeDLSCH.LDPCDecodingAlgorithm = "Normalized min-sum";
decodeDLSCH.MaximumLDPCIterationCount = 6;
decodeDLSCH.TransportBlockLength = trBlkSizes;

% Apply DL-SCH Decoder
[rx_decbits,blkerr] = decodeDLSCH(1.0 - 2*hardbits,pdsch.Modulation,pdsch.NumLayers,0);

% Results
[nErrosPayload,berPayload] = biterr(trBlk,rx_decbits);
[nErrosCR,berCR] = biterr(codedTrBlock,hardbits);

% Show Results
fprintf('BER FINAL ENTRE PAYLOAD TRANSMITIDO E RECEBIDO = %.2f, NUM ERROS = %d\n',berPayload,nErrosPayload);
fprintf('BER DA SEQUENCIA COM BITS DUPLICADOS PELO CODE RATE = %.2f, NUM ERROS = %d\n',berCR,nErrosCR);