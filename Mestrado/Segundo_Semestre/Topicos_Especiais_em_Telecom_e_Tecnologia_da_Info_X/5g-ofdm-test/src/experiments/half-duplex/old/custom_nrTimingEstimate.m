function [timingOffset, mag] = custom_nrTimingEstimate(carrier, rxWaveform, refGrid)
    % custom_nrTimingEstimate Estima o atraso temporal por correlação cruzada
    
    % Reconstrói o sinal de referência no domínio do tempo (apenas os 4 primeiros símbolos)
    [refWaveform, ~] = custom_nrOFDMModulate(carrier, refGrid);
    
    % Correlação cruzada entre o sinal recebido e a referência
    [corr, lags] = xcorr(rxWaveform, refWaveform);
    
    % Considera apenas a parte causal (lags >= 0)
    validIdx = lags >= 0;
    corr_causal = abs(corr(validIdx));
    lags_causal = lags(validIdx);
    
    % Encontra o pico de correlação
    [mag, maxIdx] = max(corr_causal);
    timingOffset = lags_causal(maxIdx);
end