function [offset, mag] = custom_nrTimingEstimate(carrier, waveform, refGrid)
    % custom_nrTimingEstimate Replica a logica do nrTimingEstimate (sintaxe CARRIER,WAVEFORM,REFGRID)
    %
    % - Modula refGrid (mesmo CP do NR) e correlaciona com cada coluna do waveform
    %   (colunas = "antenas de recepcao"; usado por nrOFDM_CFO para testar varias frequencias);
    % - Se o waveform for mais curto que a referencia, faz zero-padding;
    % - mag = |correlacao| nos lags 0..T-1 (matriz T-por-R, como no toolbox);
    % - offset = indice do pico da soma de mag nas colunas, menos 1.
    % A correlacao e feita via FFT (sem depender do xcorr do Octave).

    ref = custom_nrOFDMModulate(carrier, refGrid);
    % Como no toolbox, o waveform e T-por-Nr (colunas = antenas). Uma LINHA (1-por-N) seria lida como
    % N antenas de 1 amostra: o resultado seria sem sentido (e a memoria estoura). Falha cedo.
    if isrow(waveform) && numel(waveform) > 1
        error('custom_nrTimingEstimate:RowVector', ...
            ['waveform chegou como vetor LINHA (%dx%d); esperado COLUNA (T x Nr). ', ...
             'Provavel .'' indevido antes desta chamada.'], size(waveform,1), size(waveform,2));
    end
    [T, R] = size(waveform);
    minlength = numel(ref);
    if T < minlength
        waveformPad = [waveform; zeros(minlength - T, R)];
        T = minlength;
    else
        waveformPad = waveform;
    end

    Nc = 2^nextpow2(T + numel(ref));
    Y = conj(fft(ref, Nc));
    mag = zeros(T, R);
    for r = 1:R
        c = ifft(fft(waveformPad(:, r), Nc) .* Y);      % c[d] = sum x[n+d]*conj(ref[n])
        mag(:, r) = abs(c(1:T));
    end

    [~, peakindex] = max(sum(mag, 2));
    offset = peakindex - 1;
end
