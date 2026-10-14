% Test 5G New Radio over ADALM Pluto/2000
% Developed by Augusto Peterle
% 17/04/2023
import clib.libm2k.libm2k.*

% Add function paths
addpath('../../adalm2000');
addpath('../../adalm-pluto');
addpath('../../etc');
addpath('../../filters');
addpath('../../newradio');


%% Common Parameters
pCHDownlink = downlinkCH;           % Channel can be: 'software', or 'adalm2000'
pCHUplink = uplinkCH;               % Channel can be: 'software', or 'adalm-pluto'
pPlotFigures = true;                % True plot figures and false = no plot
pMaxIt= 10;                         % Max Iteration Loop
pMaxRetry = 3;                      % Max retry transmissions

%% gNodeB Parameters
gNBConfig.pSCS = 15;                      % Subcarrier spacing
gNBConfig.pSizeGrid = 25;                 % Size grid - Number of resource blocks
gNBConfig.pBW = 5e6;                      % Output element Bandwidth
gNBConfig.pCyclicPrefix = 'Normal';       % Cyclic Prefix 'Normal' or 'Extended'
gNBConfig.pNCellID = 102;                 % Number of Cell ID used in PBCH
gNBConfig.pLayers = 1;                    % Number of layers send over the antenna
gNBConfig.DCIModulation = dciCurrentModulation;             % DCI Modulation:16QAM coderate: 434/1024

% Check MCS LookupTable gNB Configuration
[gNBConfig.pDigitalModulation,...
    gNBConfig.pCodeRate,~] = getMCSLookupTable(gNBConfig.DCIModulation);

%% UE Parameters
UEConfig.pSCS = 15;                       % Subcarrier spacing
UEConfig.pSizeGrid = 25;                  % Size grid - Number of resource blocks
UEConfig.pBW = 5e6;                       % Output element Bandwidth
UEConfig.pCyclicPrefix = 'Normal';        % Cyclic Prefix 'Normal' or 'Extended'
UEConfig.pLayers = 1;                     % Number of layers send over the antenna
UEConfig.DCIModulation = gNBConfig.DCIModulation;             % DCI Modulation:16QAM coderate: 434/1024

% Check MCS LookupTable - At this point GNB and UE uses the same DCI
[UEConfig.pDigitalModulation,...
    UEConfig.pCodeRate,~] = getMCSLookupTable(gNBConfig.DCIModulation);

%% Software Simulation parameters
pSimArg.pSNRdB = 100;                            % SNR for AWGN Channel in dB
pSimArg.pFrequencyOffset = 1200;               % Frequency offset (Hz)
pSimArg.pPhaseOffset = 15;                      % Phase offset (Degrees)

%% ADALM PLUTO Parameters
pPlutoArg.pGain = 0;                          % Transmitter Gain in dB
pPlutoArg.pCenterFrequency = 2.4e9;             % Pass-band Center Frequency
pRadioIDTx = "ip:192.168.70.41";                % Transmitter Radio IP
pRadioIDRx = "ip:192.168.70.61";                % Receiver Radio IP

%% ADALM 2000 Parameters
pM2K.pVLCCenterFrequency = 5e6;          % VLC Center Frequency
pM2K.pDACSampleRate = 75e6;              % DAC Available Sample Rate
pM2K.pADCSampleRate = 100e6;             % ADC Available Sample Rate
pM2KTxIP = "ip:192.168.70.62";
pM2KRxIP = "ip:192.168.70.42";
%% Saved signals to generate reports

% gNodeB Time and Frequency Signals
gNBTxTime=[]; gNBRxTime=[]; gNBTxFreq=[];gNBRxFreq=[];
gNBRxBer=[]; gNBRxEVM=[];

% UE Time and Frequency Signals
UETxTime=[]; UERxTime=[]; UETxFreq=[]; UERxFreq=[];
UERxBer=[]; UERxEVM=[];

totalRetry = 0;         % Total number of retries
countOutlier = 0;       % Count outlier identified
fullDLTx = [];          % Total Bits transfered in downlink payload
fullDLRx  = [];         % Total Bits received in downlink payload
fullULTx = [];          % Total Bits transfered in uplink payload
fullULRx = [];          % Total bits received in uplink payload

fullDLTxnoCRC = [];     % Total bits received in downlink payload withou correction
fullDLRxnoCRC = [];     % Total bits received in downlink payload withou correction
fullULTxnoCRC = [];     % Total bits received in downlink payload withou correction
fullULRxnoCRC = [];     % Total bits received in downlink payload withou correction

fullDLsymEqRx=[];       % Symbols received in DL Equalized
fullDLsymnEqRx=[];      % Symbols received in DL not Equalized
fullULsymEqRx=[];       % Symbols received in UL Equalized
fullULsymnEqRx=[];      % Symbols received in UL not Equalized
fullDLTxsym=[];         % Symbols transmited in Downlink
fullULTxsym=[];         % Symbols transmited in Uplink
fullDLPLTx=[];          % Full payload transmitted without retransmission
fullDLPLRx=[];          % Full payload receiver without retransmission
berDL = 0;              % Bit error rate in Donwnlink Transmission
berUL = 0;              % Bit error rate in Uplink Transmis sion
berDLnoCRC = 0;         % Bit error rate without compensate Coding Rate in Downlink
berULnoCRC = 0;         % Bit error rate withou compensate Coding Rate in Uplink
berDLwR = 0;            % Bit error rate without retransmissions
EVM_pUL = 0;            % EVM for received uplink
EVM_pDL = 0;            % EVM for received downlink

%% Setup Objects
gNB = initGNB(gNBConfig);
UE = initUE(UEConfig);

% Hardware Setup for Downlink
if ~strcmp(pCHDownlink,'software')
    % Init ADALM 2000 devices - Run another script because it needs to
    % import some specific libraries
    initM2KSDR;
end

% Hardware Setup for Uplink
if ~strcmp(pCHUplink,'software')
    % Init Pluto RF Devices
    [gNBPluto,UEPluto] = initPlutoSDR(pRadioIDTx,pRadioIDRx,...
        pPlutoArg.pCenterFrequency,pPlutoArg.pGain,gNB.cSampleRate/100,gNB,UE);
end

%% Simulation Loop
cont = 0;               % Iteration Counter

% First payload
gNBtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
while (cont < pMaxIt)
   
    % Set Local buffers
    auxDLTxnoCRC = [];      % Auxiliar Total bits received in downlink payload withou correction in one iteration
    auxDLRxnoCRC = [];      % Auxiliar Total bits received in downlink payload withou correction in one iteration
    auxULTxnoCRC = [];      % Auxiliar Total bits received in downlink payload withou correction in one iteration
    auxULRxnoCRC = [];      % Auxiliar Total bits received in downlink payload withou correction in one iteration
    auxDLTx = [];           % Total Bits transfered in downlink payload
    auxDLRx  = [];          % Total Bits received in downlink payload
    auxULTx = [];           % Total Bits transfered in uplink payload
    auxULRx = [];           % Total bits received in uplink payload
    
    %% Send downlink Frame
    % Display Counter index
    fprintf("***************\nITERATION = %d, DCI = %d \n***************\n",cont,dciCurrentModulation);

    % Create new downlink random Frame
    fprintf(2,'Send new Downlink Frame from gNB to UE\n');
    gNBdciBits = initDCI(gNBConfig.DCIModulation,1,0);
    [txDLgrid,txDLwaveform,txDLsym,gNBCodedBlk] = initGNBDownlink(gNB,gNBConfig,...
        gNBdciBits,gNBtrBlk);

    % Transmitit Downlink via software
    if strcmp(pCHDownlink,'software')
        [txDLwaveformCH,~] = transmitFrame(pCHDownlink,-1,gNB,pSimArg,txDLwaveform,-1);
    else
        % Transmit Downlink via ADALM2000
        [txDLwaveformCH,frameSize] = transmitFrame(pCHDownlink,aout,...
            gNB,-1,txDLwaveform,pM2K);
    end

    % Receive Downlink
    if strcmp(pCHDownlink,'software')
        rxDLwaveform = txDLwaveformCH;
    else
        % Receive Downlink via ADALM2000
        % Reset Analog Input
        ain.stopAcquisition();

        % Receive Downlink with ADALM 20000
        [rxWaveform_bb,rxWaveform_pb] = receiveFrame(pCHDownlink,ain,...
            UE,frameSize,pM2K);

        % Decode the rxWaveforme in Baseband - Pass band only for spectrum
        % plot
        rxDLwaveform = rxWaveform_bb;
        rxWaveform_pb = rxWaveform_pb.';
        
    end

    % UE decodes Downlink Frame
    fprintf(2,'Receive the downlink from gNB in UE\n')
    [UErxBit,UErxDCI,UErxBlkErr,UErxSymEq,UErxSymnEq,UERxwaveform,UErxCell,dlHest,UErxRawBit] = ...
        decodeDownlink(UE,UEConfig,rxDLwaveform,pCHDownlink);

    % Current EVM Downlink
    [~, evmDLcur] = evm(getModulationLevel(gNB.pdsch.Modulation),txDLsym,UErxSymEq);
    fprintf('DOWNLINK CURRENT EVM = %.3f\n',evmDLcur);

    %% Send Uplink ACK for first transmission
    % Check reception
    if sum(UErxBlkErr) == 0
        fprintf('Reception without errors in UE\n')

        % Create new uplink ACK Frame from UE to GNB
        fprintf(2,'Send ACK Uplink from UE to gNB\n')
        UEtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
        UEuciBits = [1;1];
        [txULgrid,txULwaveform,txULsym,UECodedBlk] = initUEUplink(UE,UEConfig,UEuciBits,UEtrBlk);


        % Transmitit Uplink
        if strcmp(pCHUplink,'software')
            txULwaveformCH = transmitFrame(pCHUplink,-1,UE,pSimArg,txULwaveform,-1);
        else

            % Reset Pluto Transmission
            resetPluto(UEPluto,1);
            UEPluto.EnableTransmitRepeat = true;

            % Transmit Uplink via ADALM-PLUTO
            [txULwaveformCH,frameSize] = transmitFrame(pCHUplink,...
                UEPluto,UE,-1,txULwaveform,-1);
        end

        % Receive Uplink
        if strcmp(pCHUplink,'software')
            rxULwaveform = txULwaveformCH;
        else
            % Receive Uplink via ADALM-PLUTO
            resetPluto(gNBPluto,1);
            gNBPluto.EnableBurstMode = true;
            [rxWaveform_bb,~] = receiveFrame(pCHUplink,gNBPluto,UE,frameSize,-1);
            rxULwaveform = rxWaveform_bb;
            resetPluto(gNBPluto,1);
        end

        % gNB decodes Uplink Frame
        fprintf(2,'Receive uplink from UE in gNB\n')
        [gNBrxBit,gNBrxUCI,gNBBlkErr,gNBrxSymEq,gNBrxSymnEq,gNBrxwaveform,upHest,gNBRxRawBit] = ...
            decodeUplink(gNB,gNBConfig,rxULwaveform,pCHUplink);

        % Received ACK then update the payload Transport Block
        if gNBrxUCI == [1;1]
            fprintf('Decode ACK in gNB\n')
            fprintf('Update payload Transport Block\n')
            
            % Current EVM Uplink
            [~, evmULcur] = evm(getModulationLevel(UE.pusch.Modulation),txULsym,gNBrxSymEq);
            fprintf('UPLINK CURRENT EVM = %.3f\n',evmULcur);
        end
        
        % Concatenate bits
        auxDLTx = [auxDLTx;reshape(gNBtrBlk,1,[])];
        auxDLRx  = [auxDLRx ;reshape(UErxBit,1,[])];
        auxULTx = [auxULTx ; reshape(UEtrBlk,1,[])];
        auxULRx = [auxULRx ; reshape(gNBrxBit,1,[])];

        % First Try Downlink Send and received/ Uplink Send and received
        % ack
        % Counting blks send and receiveds without Coding Rate
        auxDLTxnoCRC = [auxDLTxnoCRC;reshape(gNBCodedBlk,1,[])];
        auxDLRxnoCRC  = [auxDLRxnoCRC ;reshape(UErxRawBit,1,[])];
        auxULTxnoCRC = [auxULTxnoCRC ; reshape(UECodedBlk,1,[])];
        auxULRxnoCRC = [auxULRxnoCRC ; reshape(gNBRxRawBit,1,[])];
    else

        %% Send downlink NACK for first transmission
        % Create new uplink NACK Frame from UE to GNB
        fprintf(2,'Send NACK Uplink from UE to gNB\n')
        UEtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
        UEuciBits = [1;0];
        [txULgrid,txULwaveform,txULsym,UECodedBlk] = initUEUplink(UE,UEConfig,UEuciBits,UEtrBlk);


        % Transmitit Uplink
        if strcmp(pCHUplink,'software')
            txULwaveformCH = transmitFrame(pCHUplink,-1,UE,pSimArg,txULwaveform,-1);
        else

            % Reset Pluto Transmission
            resetPluto(UEPluto,1);
            UEPluto.EnableTransmitRepeat = true;

            % Transmit Uplink via ADALM-PLUTO
            [txULwaveformCH,frameSize] = transmitFrame(pCHUplink,...
                UEPluto,UE,-1,txULwaveform,-1);
        end

        % Receive Uplink
        if strcmp(pCHUplink,'software')
            rxULwaveform = txULwaveformCH;
        else
            % Receive Uplink via ADALM-PLUTO
            resetPluto(gNBPluto,1);
            gNBPluto.EnableBurstMode = true;
            [rxWaveform_bb,~] = receiveFrame(pCHUplink,gNBPluto,UE,frameSize,-1);
            rxULwaveform = rxWaveform_bb;
            resetPluto(gNBPluto,1);
        end

        % gNB decodes Uplink Frame
        fprintf(2,'Receive uplink from UE in gNB\n')
        [gNBrxBit,gNBrxUCI,gNBBlkErr,gNBrxSymEq,gNBrxSymnEq,gNBrxwaveform,upHest,gNBRxRawBit] = ...
            decodeUplink(gNB,gNBConfig,rxULwaveform,pCHUplink);

        % Received NACK then update the payload Transport Block
        if gNBrxUCI ~= [1;1]
            fprintf('Decode NACK in gNB\n')
            fprintf('Update payload Transport Block\n')
        end

        % Concatenate bits
        auxDLTx = [auxDLTx;reshape(gNBtrBlk,1,[])];
        auxDLRx  = [auxDLRx ;reshape(UErxBit,1,[])];
        auxULTx = [auxULTx ; reshape(UEtrBlk,1,[])];
        auxULRx = [auxULRx ; reshape(gNBrxBit,1,[])];

        % First Try Downlink Send and received/ Uplink Send and received
        % nack
        % Counting blks send and receiveds without CRC
        auxDLTxnoCRC = [auxDLTxnoCRC;reshape(gNBCodedBlk,1,[])];
        auxDLRxnoCRC  = [auxDLRxnoCRC ;reshape(UErxRawBit,1,[])];
        auxULTxnoCRC = [auxULTxnoCRC ; reshape(UECodedBlk,1,[])];
        auxULRxnoCRC = [auxULRxnoCRC ; reshape(gNBRxRawBit,1,[])];

        % Current EVM Uplink
        [~, evmULcur] = evm(getModulationLevel(UE.pusch.Modulation),txULsym,gNBrxSymEq);
        fprintf('UPLINK CURRENT EVM = %.3f\n',evmULcur);

        % Received transmission with errors
        retryCounter = 0;

        % Retry until success or pMaxRetry
        while (retryCounter < pMaxRetry)

            %% Send downlink retransmission
            % Retransmit downlink Frame
            fprintf(2,'Retry number %d Downlink Frame from gNB to UE\n',retryCounter);
            gNBdciBits = initDCI(gNBConfig.DCIModulation,1,0);
            [txDLgrid,txDLwaveform,txDLsym,gNBCodedBlk] = initGNBDownlink(gNB,gNBConfig,...
                gNBdciBits,gNBtrBlk);

            % Transmitit Downlink via software
            if strcmp(pCHDownlink,'software')
                [txDLwaveformCH,~] = transmitFrame(pCHDownlink,-1,gNB,pSimArg,txDLwaveform,-1);

            else
                % Transmit Downlink via ADALM2000
                [txDLwaveformCH,frameSize] = transmitFrame('adalm2000',aout,...
                    gNB,-1,txDLwaveform,pM2K);
            end

            % Receive Downlink
            if strcmp(pCHDownlink,'software')
                rxDLwaveform = txDLwaveformCH;
            else
                % Reset Analog Input
                ain.stopAcquisition();

                % Receive Downlink with ADALM 20000
                [rxWaveform_bb,rxWaveform_pb] = receiveFrame(pCHDownlink,ain,...
                    UE,frameSize,pM2K);

                % Decode the rxWaveforme in Baseband - Pass band only for spectrum
                rxDLwaveform = rxWaveform_bb;
                rxWaveform_pb = rxWaveform_pb.';
            end

            % UE decodes Downlink Frame
            fprintf(2,'Receive the retry downlink %d downlink from gNB in UE\n',retryCounter);
            [UErxBit,UErxDCI,UErxBlkErr,UErxSymEq,UErxSymnEq,UERxwaveform,UErxCell,dlHest,UErxRawBit] = ...
                decodeDownlink(UE,UEConfig,rxDLwaveform,pCHDownlink);

            % Current EVM Downlink
            [~, evmDLcur] = evm(getModulationLevel(gNB.pdsch.Modulation),txDLsym,UErxSymEq);
            fprintf('DOWNLINK CURRENT EVM = %.3f\n',evmDLcur);

            %% Send uplink ACK for retransmission
            % Check reception
            if sum(UErxBlkErr) == 0

                % Display success reception in downlink retransmission
                fprintf('Retransmission Reception without errors in UE\n')

                % Create new uplink ACK Frame from UE to GNB
                fprintf(2,'Send retry ACK %d Uplink from UE to gNB\n',retryCounter)
                UEtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
                UEuciBits = [1;1];
                [txULgrid,txULwaveform,txULsym,UECodedBlk] = initUEUplink(UE,UEConfig,UEuciBits,UEtrBlk);

                % Transmitit Uplink
                if strcmp(pCHUplink,'software')
                    txULwaveformCH = transmitFrame(pCHUplink,-1,UE,pSimArg,txULwaveform,-1);
                else

                    % Transmit Uplink via ADALM-PLUTO
                    resetPluto(UEPluto,1);
                    [txULwaveformCH,frameSize] = transmitFrame(pCHUplink,...
                        UEPluto,UE,-1,txULwaveform,-1);
                end

                % Receive Uplink
                if strcmp(pCHUplink,'software')
                    rxULwaveform = txULwaveformCH;
                else

                    % Receive Uplink via ADALM-PLUTO
                    resetPluto(gNBPluto,1);
                    [rxWaveform_bb,~] = receiveFrame(pCHUplink,gNBPluto,UE,frameSize,-1);
                    rxULwaveform = rxWaveform_bb;
                    resetPluto(gNBPluto,1);
                end

                % Decode uplink
                % gNB receives Uplink Frame
                fprintf(2,'Receive retry uplink %d from UE in gNB\n',retryCounter)
                [gNBrxBit,gNBrxUCI,gNBBlkErr,gNBrxSymEq,gNBrxSymnEq,gNBrxwaveform,upHest,gNBRxRawBit] = ...
                    decodeUplink(gNB,gNBConfig,rxULwaveform,pCHUplink);

                % Received ACK then update the payload Transport Block
                if gNBrxUCI == [1;1]
                    fprintf('Decode ACK in gNB retry %d\n',retryCounter)
                    fprintf('Update payload Transport Block\n')
                    retryCounter = 1000;
                    
                    % Current EVM Uplink
                    [~, evmULcur] = evm(getModulationLevel(UE.pusch.Modulation),txULsym,gNBrxSymEq);
                    fprintf('UPLINK CURRENT EVM = %.3f\n',evmULcur);
                end
                
                % Concatenate bits
                auxDLTx = [auxDLTx;reshape(gNBtrBlk,1,[])];
                auxDLRx  = [auxDLRx ;reshape(UErxBit,1,[])];
                auxULTx = [auxULTx ; reshape(UEtrBlk,1,[])];
                auxULRx = [auxULRx ; reshape(gNBrxBit,1,[])];

                % Concatenate bits withou CRC
                % Retransmission downlink and uplink send ACK
                auxDLTxnoCRC = [auxDLTxnoCRC;reshape(gNBCodedBlk,1,[])];
                auxDLRxnoCRC  = [auxDLRxnoCRC ;reshape(UErxRawBit,1,[])];
                auxULTxnoCRC = [auxULTxnoCRC ; reshape(UECodedBlk,1,[])];
                auxULRxnoCRC = [auxULRxnoCRC ; reshape(gNBRxRawBit,1,[])];
            else
                
                %% Send uplink NACK for transmission
                % Create new uplink NACK Frame from UE to GNB
                fprintf(2,'Send NACK Uplink from UE to gNB\n')
                UEtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
                UEuciBits = [1;0];
                [txULgrid,txULwaveform,txULsym,UECodedBlk] = initUEUplink(UE,UEConfig,UEuciBits,UEtrBlk);


                % Transmitit Uplink
                if strcmp(pCHUplink,'software')
                    txULwaveformCH = transmitFrame(pCHUplink,-1,UE,pSimArg,txULwaveform,-1);
                else

                    % Reset Pluto Transmission
                    resetPluto(UEPluto,1);
                    UEPluto.EnableTransmitRepeat = true;

                    % Transmit Uplink via ADALM-PLUTO
                    [txULwaveformCH,frameSize] = transmitFrame(pCHUplink,...
                        UEPluto,UE,-1,txULwaveform,-1);
                end

                % Receive Uplink
                if strcmp(pCHUplink,'software')
                    rxULwaveform = txULwaveformCH;
                else
                    % Receive Uplink via ADALM-PLUTO
                    resetPluto(gNBPluto,1);
                    gNBPluto.EnableBurstMode = true;
                    [rxWaveform_bb,~] = receiveFrame(pCHUplink,gNBPluto,UE,frameSize,-1);
                    rxULwaveform = rxWaveform_bb;
                    resetPluto(gNBPluto,1);
                end

                % gNB decodes Uplink Frame
                fprintf(2,'Receive uplink from UE in gNB\n')
                [gNBrxBit,gNBrxUCI,gNBBlkErr,gNBrxSymEq,gNBrxSymnEq,gNBrxwaveform,upHest,gNBRxRawBit] = ...
                    decodeUplink(gNB,gNBConfig,rxULwaveform,pCHUplink);

                % Received NACK then update the payload Transport Block
                if gNBrxUCI ~= [1;1]
                    fprintf('Decode NACK in gNB\n')
                    fprintf('Update payload Transport Block\n')
                end
                
                % Current EVM Uplink
                [~, evmULcur] = evm(getModulationLevel(UE.pusch.Modulation),txULsym,gNBrxSymEq);
                fprintf('UPLINK CURRENT EVM = %.3f\n',evmULcur);
                
                % Concatenate bits
                auxDLTx = [auxDLTx;reshape(gNBtrBlk,1,[])];
                auxDLRx  = [auxDLRx ;reshape(UErxBit,1,[])];
                auxULTx = [auxULTx ; reshape(UEtrBlk,1,[])];
                auxULRx = [auxULRx ; reshape(gNBrxBit,1,[])];

                % Concatenate bits withou CRC
                % Retransmission downlink and uplink send ACK
                auxDLTxnoCRC = [auxDLTxnoCRC;reshape(gNBCodedBlk,1,[])];
                auxDLRxnoCRC  = [auxDLRxnoCRC ;reshape(UErxRawBit,1,[])];
                auxULTxnoCRC = [auxULTxnoCRC ; reshape(UECodedBlk,1,[])];
                auxULRxnoCRC = [auxULRxnoCRC ; reshape(gNBRxRawBit,1,[])];
                
                
            end

            % Update retryCounter
            retryCounter = retryCounter + 1;
            totalRetry = totalRetry + 1;
        end

    end
    
    % Concatenate payload data without retransmissions
    fullDLPLTx=[fullDLPLTx ; reshape(gNBtrBlk,1,[]) ]; 
    fullDLPLRx=[fullDLPLRx ; reshape(UErxBit,1,[])  ];
    
    %% Check Loop data consistence
    if strcmp(pCHUplink,'software') && strcmp(pCHDownlink,'software') 
        cont = cont + 1;

        % Concatenate Transport blocks
        fullDLTx = [fullDLTx ; auxDLTx];
        fullDLRx  = [fullDLRx ; auxDLRx];
        fullULTx = [fullULTx ; auxULTx];
        fullULRx = [fullULRx ; auxULRx];
        
        % Concatenate bits transferred without CR
        fullDLTxnoCRC = [fullDLTxnoCRC ; auxDLTxnoCRC];
        fullDLRxnoCRC  = [fullDLRxnoCRC ; auxDLRxnoCRC];
        fullULTxnoCRC = [fullULTxnoCRC; auxULTxnoCRC];
        fullULRxnoCRC = [fullULRxnoCRC ; auxULRxnoCRC ];

        % Concatenate Downlink Symbols
        fullDLTxsym=[fullDLTxsym;txDLsym];
        fullDLsymEqRx=[fullDLsymEqRx;UErxSymEq];
        fullDLsymnEqRx=[fullDLsymnEqRx;UErxSymnEq];

        % Concatenate Uplink Symbols
        fullULTxsym=[fullULTxsym;txULsym];
        fullULsymEqRx=[fullULsymEqRx;gNBrxSymEq];
        fullULsymnEqRx=[fullULsymnEqRx;gNBrxSymnEq];

        % Update transport Block
        gNBtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
    end

    if (evmULcur < 7.5 && strcmp(pCHUplink,'adalm-pluto') && strcmp(pCHDownlink,'software') )
        cont = cont + 1;

        % Concatenate Transport blocks
        fullDLTx = [fullDLTx ; auxDLTx];
        fullDLRx  = [fullDLRx ; auxDLRx];
        fullULTx = [fullULTx ; auxULTx];
        fullULRx = [fullULRx ; auxULRx];
        
        % Concatenate bits transferred without CR
        fullDLTxnoCRC = [fullDLTxnoCRC ; auxDLTxnoCRC];
        fullDLRxnoCRC  = [fullDLRxnoCRC ; auxDLRxnoCRC];
        fullULTxnoCRC = [fullULTxnoCRC; auxULTxnoCRC];
        fullULRxnoCRC = [fullULRxnoCRC ; auxULRxnoCRC ];

        % Concatenate Downlink Symbols
        fullDLTxsym=[fullDLTxsym;txDLsym];
        fullDLsymEqRx=[fullDLsymEqRx;UErxSymEq];
        fullDLsymnEqRx=[fullDLsymnEqRx;UErxSymnEq];

        % Concatenate Uplink Symbols
        fullULTxsym=[fullULTxsym;txULsym];
        fullULsymEqRx=[fullULsymEqRx;gNBrxSymEq];
        fullULsymnEqRx=[fullULsymnEqRx;gNBrxSymnEq];

        % Update transport Block
        gNBtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
    end

    if (strcmp(pCHUplink,'software') && evmDLcur < 10 && strcmp(pCHDownlink,'adalm2000') )
        cont = cont + 1;

        % Concatenate Transport blocks
        fullDLTx = [fullDLTx ; auxDLTx];
        fullDLRx  = [fullDLRx ; auxDLRx];
        fullULTx = [fullULTx ; auxULTx];
        fullULRx = [fullULRx ; auxULRx];
        
        % Concatenate bits transferred without CR
        fullDLTxnoCRC = [fullDLTxnoCRC ; auxDLTxnoCRC];
        fullDLRxnoCRC  = [fullDLRxnoCRC ; auxDLRxnoCRC];
        fullULTxnoCRC = [fullULTxnoCRC; auxULTxnoCRC];
        fullULRxnoCRC = [fullULRxnoCRC ; auxULRxnoCRC ];

        % Concatenate Downlink Symbols
        fullDLTxsym=[fullDLTxsym;txDLsym];
        fullDLsymEqRx=[fullDLsymEqRx;UErxSymEq];
        fullDLsymnEqRx=[fullDLsymnEqRx;UErxSymnEq];

        % Concatenate Uplink Symbols
        fullULTxsym=[fullULTxsym;txULsym];
        fullULsymEqRx=[fullULsymEqRx;gNBrxSymEq];
        fullULsymnEqRx=[fullULsymnEqRx;gNBrxSymnEq];

        % Update transport Block
        gNBtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
    end
    
    if (evmULcur < 5 && strcmp(pCHUplink,'adalm-pluto') && evmDLcur < evmDLmax && evmDLcur > evmDLmin && strcmp(pCHDownlink,'adalm2000') )
        cont = cont + 1;
        countOutlier = 0;
        % Concatenate Transport blocks
        fullDLTx = [fullDLTx ; auxDLTx];
        fullDLRx  = [fullDLRx ; auxDLRx];
        fullULTx = [fullULTx ; auxULTx];
        fullULRx = [fullULRx ; auxULRx];
        
        % Concatenate bits transferred without CR
        fullDLTxnoCRC = [fullDLTxnoCRC ; auxDLTxnoCRC];
        fullDLRxnoCRC  = [fullDLRxnoCRC ; auxDLRxnoCRC];
        fullULTxnoCRC = [fullULTxnoCRC; auxULTxnoCRC];
        fullULRxnoCRC = [fullULRxnoCRC ; auxULRxnoCRC ];

        % Concatenate Downlink Symbols
        fullDLTxsym=[fullDLTxsym;txDLsym];
        fullDLsymEqRx=[fullDLsymEqRx;UErxSymEq];
        fullDLsymnEqRx=[fullDLsymnEqRx;UErxSymnEq];

        % Concatenate Uplink Symbols
        fullULTxsym=[fullULTxsym;txULsym];
        fullULsymEqRx=[fullULsymEqRx;gNBrxSymEq];
        fullULsymnEqRx=[fullULsymnEqRx;gNBrxSymnEq];

        % Update transport Block
        gNBtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
    else
        % Update gNB Transport block to avoid freezing
        gNBtrBlk = randi([0 1],gNB.trBlkSizes,gNB.cCarrier.SlotsPerFrame-1);
    end
    
    % Detect Outlier in Downlink
    if ( (evmDLcur > evmDLmax || evmDLcur < evmDLmin) && strcmp(pCHDownlink,'adalm2000'))
        countOutlier = countOutlier + 1;
    end
    
    % If several outliers occurs then recalibrate DAC and ADC
    if countOutlier > 2
        
        %Stop
        ain.stopAcquisition();
        aout.stop();
        
        % Calibration gNB
        fprintf('Closing current context');
        clib.libm2k.libm2k.context.contextCloseAll();
        initM2KSDR;
        
        % Reset outlier counter
        countOutlier = 0;
    end


    %% Results

    % Bit Error Rate - BER
    if ~isempty(fullDLRx) && ~isempty(fullULRx)
        [nErrosDL,berDL] = biterr(fullDLTx,fullDLRx );
        [nErrosUL,berUL] = biterr(fullULTx,fullULRx );
        [nErrosDLnoCRC,berDLnoCRC] = biterr(fullDLTxnoCRC,fullDLRxnoCRC);
        [nErrosULnoCRC,berULnoCRC] = biterr(fullULTxnoCRC,fullULRxnoCRC);
        [nErrosDLwR,berDLwR] = biterr(fullDLPLTx,fullDLPLRx );
    end

    % Error Vector Magnitude - EVM
    if ~isempty(fullULsymEqRx) && ~isempty(fullDLsymEqRx)
        [EVM_dBUL, EVM_pUL] = evm(16,fullULTxsym,fullULsymEqRx);
        [EVM_dBDL, EVM_pDL] = evm(16,fullDLTxsym,fullDLsymEqRx);
    end

    % Update results Dashboard
    updateTimeDashboard(txDLwaveform,UERxwaveform,txULwaveform,gNBrxwaveform,1/gNB.cSampleRate);
    if ~strcmp(pCHDownlink,'software')
        updateFreqDashboard(rxWaveform_pb,gNBrxwaveform,1/pM2K.pDACSampleRate,1/gNB.cSampleRate)
    else
        updateFreqDashboard(UERxwaveform,gNBrxwaveform,1/UE.cSampleRate,1/gNB.cSampleRate);
    end
    updateCnstDashboard(fullDLsymnEqRx,fullDLsymEqRx,fullULsymnEqRx,fullULsymEqRx);
    updateGridDashboard(txDLgrid,txULgrid);
    updateAnnotDashboard(berDL,berUL,EVM_pDL,EVM_pUL,cont,totalRetry);
    updateAnnotDashboard2(berDLnoCRC,berULnoCRC,berDLwR);
    updateChannelDashboard(dlHest,upHest);
    pause(1)

end

% Close adalm 2000 contexts
clib.libm2k.libm2k.context.contextCloseAll();
clear m2k

if(strcmp(pCHDownlink,'adalm2000'))
    clear gNBM2k
end

if(strcmp(pCHUplink,'adalm2000'))
    clear UEM2k
end
