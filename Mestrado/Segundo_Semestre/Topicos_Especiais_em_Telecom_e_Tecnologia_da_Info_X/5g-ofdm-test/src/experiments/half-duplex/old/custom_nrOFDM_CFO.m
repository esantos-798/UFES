function [correctedWaveform, appliedFrequencyCorrection] = custom_nrOFDM_CFO(rxWaveform, frequencyCorrectionRange, refGrid, sampleRate, carrier)
    % custom_nrOFDM_CFO Correção de CFO via varredura de frequência
    
    t = (0:length(rxWaveform)-1)' / sampleRate;
    maxMetric = -1;
    bestCFO = 0;
    
    % Modula a grade de referência para o domínio do tempo
    [refWaveform, ~] = custom_nrOFDMModulate(carrier, refGrid);
    refLen = length(refWaveform);
    
    if refLen > length(rxWaveform)
        refLen = length(rxWaveform);
        refWaveform = refWaveform(1:refLen);
    end

    % Varredura na faixa de correção
    for f_test = frequencyCorrectionRange
        % Testa a correção de frequência
        testWaveform = rxWaveform .* exp(-1i * 2 * pi * f_test * t);
        
        % Avalia a energia da correlação com o sinal de referência
        metric = abs(sum(testWaveform(1:refLen) .* conj(refWaveform)));
        
        if metric > maxMetric
            maxMetric = metric;
            bestCFO = f_test;
        end
    end

    appliedFrequencyCorrection = bestCFO;
    correctedWaveform = rxWaveform .* exp(-1i * 2 * pi * appliedFrequencyCorrection * t);
end