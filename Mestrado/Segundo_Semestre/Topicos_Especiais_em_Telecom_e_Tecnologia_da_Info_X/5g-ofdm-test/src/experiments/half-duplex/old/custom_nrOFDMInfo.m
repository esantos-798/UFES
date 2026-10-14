function info = custom_nrOFDMInfo(carrier)
    % custom_nrOFDMInfo Emula a extração de informações da portadora 5G NR
    % 
    % Calcula dinamicamente o tamanho da FFT (Nfft) e a taxa de amostragem
    % (SampleRate) baseando-se no SubcarrierSpacing e no NSizeGrid.

    % No 5G NR, cada Resource Block (RB) possui 12 subportadoras.
    active_sc = carrier.NSizeGrid * 12;

    % Define o tamanho da FFT para a potência de 2 mais próxima 
    % que comporte as subportadoras ativas com margem para as bandas de guarda.
    if active_sc <= 300
        info.Nfft = 512;
    elseif active_sc <= 600
        info.Nfft = 1024;
    elseif active_sc <= 1200
        info.Nfft = 2048;
    else
        info.Nfft = 4096;
    end

    % A taxa de amostragem no 5G é dada por: SCS * Nfft
    info.SampleRate = carrier.SubcarrierSpacing * 1e3 * info.Nfft;
end
