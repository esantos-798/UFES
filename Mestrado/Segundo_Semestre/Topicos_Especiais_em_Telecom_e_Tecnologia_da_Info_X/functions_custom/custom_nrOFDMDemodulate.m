function grid = custom_nrOFDMDemodulate(carrier, waveform, varargin)
    % custom_nrOFDMDemodulate Emula nrOFDMDemodulate
    %
    % Como no toolbox, a janela da FFT e posicionada em CyclicPrefixFraction
    % (default 0.5) do CP e a rampa de fase resultante e compensada. Isso da
    % tolerancia a erro de timing "adiantado" de ate metade do CP.
    % Opcoes: 'CyclicPrefixFraction'. Outras (ex.: 'SampleRate') sao ignoradas.

    frac = 0.5;
    for i = 1:2:numel(varargin)
        if ischar(varargin{i}) && strcmpi(varargin{i}, 'CyclicPrefixFraction')
            frac = varargin{i+1};
        end
    end

    info = custom_nrOFDMInfo(carrier);
    Nfft = info.Nfft;
    K = carrier.NSizeGrid * 12;
    waveform = waveform(:);

    maxL = floor(numel(waveform) / Nfft);
    cpAll = custom_nrSymbolLengths(carrier, info, maxL);
    cumlen = cumsum(cpAll + Nfft);
    L = sum(cumlen <= numel(waveform));            % simbolos completos
    starts = [0, cumlen(1:max(L-1,0))];

    start_idx = floor(Nfft/2 - K/2) + 1;
    k = (-Nfft/2 : Nfft/2-1).';                    % indice de frequencia apos fftshift
    grid = zeros(K, L);
    for l = 1:L
        cp = cpAll(l);
        sh = round(frac * cp);
        seg = waveform(starts(l) + cp - sh + 1 : starts(l) + cp - sh + Nfft);
        X = fftshift(fft(seg, Nfft)) / sqrt(Nfft);
        X = X .* exp(1i * 2*pi * k * sh / Nfft);   % compensa o deslocamento da janela
        grid(:, l) = X(start_idx:start_idx+K-1);
    end
end
