function [correctedWaveform,appliedFrequencyCorrection] = nrOFDM_CFO(waveform,frequencyCorrectionRange,refGrid,sampleRate,carrier)

% Apply frequency offsets to the waveform as specified by
% Input Parameters
% waveform - waveform to correct frequency shifts
% frequencyCorrectioRange - vector with Coarse or Fine Frequency vector
% refGrid - Reference Grid of PSS 
% sampleRate - waveform samplerate
% carrier - 5G NR carrier structure

% Output Parameters
% correctedWaveform - waveform with frequency correction
% appliefFrequencyCorrection - Frequency offset calculated

% Number of samples to create a new tone 
nSamples = (0:length(waveform)-1)';
frequencyShift = (2*pi*frequencyCorrectionRange.*nSamples)./sampleRate;

% Each column represents an offset waveform.
offsetWaveforms = waveform.*exp(1j*frequencyShift);

% Cross-correlation with frequency shifted waveforms and ref PSS Grid
[~,mag] = nrTimingEstimate(carrier,offsetWaveforms,refGrid);

% Find the frequency at which the PSS correlation is at a maximum.
[~,index] = max(max(mag));    
appliedFrequencyCorrection = frequencyCorrectionRange(index);
correctedWaveform = offsetWaveforms(:,index);
end

