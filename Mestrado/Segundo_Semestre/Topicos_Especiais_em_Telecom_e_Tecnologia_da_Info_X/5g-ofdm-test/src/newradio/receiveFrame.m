function [rxWaveform_bb,rxWaveform_pb] = receiveFrame(pChannel,pDevice,txStructure,frameSize,pM2K)
%RECEIVEFRAME Summary of this function goes here
%   Detailed explanation goes here

if strcmp(pChannel,'adalm-pluto')
   
    % Receive a ADALM Pluto Frame
    [rxWaveform_bb,datavalid,overflow] = pDevice();
    if ~(overflow) % no dropped samples
        if ~(datavalid) % received desired data
             disp('Received INVALID Data ADALM-PLUTO');
        end
        disp('Received VALID Data ADALM-PLUTO');
    end
    
    rxWaveform_bb = rxWaveform_bb - mean(rxWaveform_bb);
    rxWaveform_bb = rxWaveform_bb/max(abs(rxWaveform_bb));
    rxWaveform_pb = rxWaveform_bb;

elseif strcmp(pChannel,'adalm2000')
    %% VLC Channel with ADALM 2000
    
    % Read Analog Input
    rxWaveform_pb = pDevice.getSamplesInterleaved_matlab(round(5.5*frameSize));
    rxWaveform_pb = rxWaveform_pb.double;
    
    % Normalize and remove DC of the received signal
    rxWaveform_pb = rxWaveform_pb - mean(rxWaveform_pb);
    rxWaveform_pb = rxWaveform_pb/max(abs(rxWaveform_pb));
    
    
    % Resample the received signal to match ADC to DAC Sampling Rate
    [P1, Q1] = rat(pM2K.pADCSampleRate/(pM2K.pDACSampleRate/2));
    %[P1, Q1] = rat(pM2K.pADCSampleRate/(pM2K.pDACSampleRate));
    rxWaveform_pb = resample(rxWaveform_pb,Q1,P1);
       
    % Demodulate Back to the Baseband
    [rxReal,rxImag] = demod(rxWaveform_pb,...
        pM2K.pVLCCenterFrequency,pM2K.pDACSampleRate,'qam');
    rxWaveform_up = rxReal + 1i.*rxImag;
    
    % Downsample
    [P1bb, Q1bb] = rat(pM2K.pDACSampleRate/txStructure.cInfo.SampleRate);
    rxWaveform_bb = resample(rxWaveform_up,Q1bb,P1bb);
    
    % Convert to column vector
    rxWaveform_bb = rxWaveform_bb.';

    rxWaveform_bb = rxWaveform_bb - mean(rxWaveform_bb);
    rxWaveform_bb = rxWaveform_bb/max(abs(rxWaveform_bb));

    pause(0);
end

