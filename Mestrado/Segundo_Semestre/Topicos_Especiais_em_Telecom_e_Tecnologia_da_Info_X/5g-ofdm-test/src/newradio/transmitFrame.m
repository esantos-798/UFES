function [txWaveform,frameSize] = transmitFrame(pChannel,pDevice,txStructure,pSimArg,waveform,pM2K)
%TRANSMITFRAME Summary of this function goes here
%   Detailed explanation goes here

if strcmp(pChannel,'software')
    %% Software simulation
    % Add random offset with noise
    offset = randi([0 1e2]); 
    txWaveform = [zeros(offset,1);waveform];
    
    % Add white Gaussian noise to the waveform. Note that the SNR only
    txWaveform  = awgn(txWaveform,pSimArg.pSNRdB,-10*log10(double(txStructure.cInfo.Nfft)));
    
    % Introduce phase and frequency offsets
    PFO = comm.PhaseFrequencyOffset( ...
    'PhaseOffset',     pSimArg.pPhaseOffset, ...
    'FrequencyOffset', pSimArg.pFrequencyOffset,...
    'SampleRate',      txStructure.cInfo.SampleRate);
    
    % Insert Phase and Frequency Offset
    txWaveform  = PFO(txWaveform);
    
elseif strcmp(pChannel,'adalm-pluto')
    %% RF Channel - ADALM PLUTO
      
    % Transmit pluto Data
    % This run in a separate embedded thread
    pDevice.transmitRepeat(waveform);
    % Transmitted waveform
    txWaveform = waveform;
    
    
elseif strcmp(pChannel,'adalm2000')
    %% VLC Channel with ADALM 2000

    % Upsample the baseband OFDM symbol to the DAC Sample Rate
    [P1bb, Q1bb] = rat(pM2K.pDACSampleRate/txStructure.cInfo.SampleRate);
    waveform_up = resample(waveform,P1bb,Q1bb);
    
    % Apply DUC - Digital Upconverter - Analogical QAM
    txWaveform = modulate(real(waveform_up),...
        pM2K.pVLCCenterFrequency,...
        pM2K.pDACSampleRate,'qam',imag(waveform_up));
    
    %txWaveform = 3*txWaveform;
    % Write in ADALM 2000 in channel 1
    pDevice.pushInterleaved(txWaveform,1);
    
end
frameSize = length(txWaveform);
end

