function [waveform, info] = custom_nrOFDMModulate(carrier, grid, varargin)
    % custom_nrOFDMModulate Emula nrOFDMModulate (IFFT + CP normal do NR)
    %
    % Opcoes Nome-Valor (ex.: 'Windowing') sao aceitas e ignoradas; o
    % script original usa Windowing = 0, que corresponde a esta emulacao.

    info = custom_nrOFDMInfo(carrier);
    Nfft = info.Nfft;
    [K, L, P] = size(grid);
    if P > 1
        error('custom_nrOFDMModulate:Ports', 'Apenas 1 porta e suportada.');
    end

    % Subportadoras em torno do DC (indice K/2 -> DC, como no NR)
    start_idx = floor(Nfft/2 - K/2) + 1;
    fft_grid = zeros(Nfft, L);
    fft_grid(start_idx:start_idx+K-1, :) = grid;

    time_syms = ifft(ifftshift(fft_grid, 1), Nfft, 1) * sqrt(Nfft);

    cp = custom_nrSymbolLengths(carrier, info, L);
    waveform = zeros(sum(cp) + L*Nfft, 1);
    pos = 0;
    for l = 1:L
        seg = [time_syms(Nfft-cp(l)+1:Nfft, l); time_syms(:, l)];
        waveform(pos+1:pos+numel(seg)) = seg;
        pos = pos + numel(seg);
    end
end
