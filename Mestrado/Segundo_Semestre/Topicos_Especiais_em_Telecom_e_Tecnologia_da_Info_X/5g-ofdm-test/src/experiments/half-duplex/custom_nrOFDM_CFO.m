function [correctedWaveform, appliedFrequencyCorrection] = custom_nrOFDM_CFO(waveform, frequencyCorrectionRange, refGrid, sampleRate, carrier)
    % custom_nrOFDM_CFO Correcao de CFO: replica o nrOFDM_CFO.m do laboratorio.
    %
    % Para cada frequencia candidata aplica exp(+j*2*pi*f*n/Fs) (a frequencia retornada e a
    % que foi SOMADA ao sinal, ou seja, tem sinal oposto ao CFO: CFO de +1200 Hz -> -1200 Hz),
    % correlaciona cada versao com a grade de referencia (custom_nrTimingEstimate, todos os
    % lags) e escolhe a frequencia de maior pico.
    %
    % Entrada/saida identicas ao original.

    % Como no original, o waveform deve ser COLUNA (o original falharia com linha: 1xN .* Nxnf).
    if isrow(waveform) && numel(waveform) > 1
        error('custom_nrOFDM_CFO:RowVector', 'waveform chegou como vetor LINHA (1x%d); esperado COLUNA.', numel(waveform));
    end
    frequencyCorrectionRange = frequencyCorrectionRange(:).';
    nSamples = (0:length(waveform)-1)';
    frequencyShift = (2*pi*frequencyCorrectionRange .* nSamples) ./ sampleRate;
    offsetWaveforms = waveform .* exp(1j*frequencyShift);          % cada coluna = um offset

    [~, mag] = custom_nrTimingEstimate(carrier, offsetWaveforms, refGrid);

    [~, index] = max(max(mag));
    appliedFrequencyCorrection = frequencyCorrectionRange(index);
    correctedWaveform = offsetWaveforms(:, index);
end
