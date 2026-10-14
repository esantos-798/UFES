function info = custom_nrOFDMInfo(carrier)
    % custom_nrOFDMInfo Emula nrOFDMInfo (CP normal)
    %
    % Nfft: menor potencia de 2 >= K/0.85 (minimo 128), onde K = 12*NSizeGrid.
    % Para 25 RB / 15 kHz resulta Nfft = 512 e SampleRate = 7.68 MHz.
    % (Regra reconstruida; exata para o caso do script. Para outras
    % larguras, confira com nrOFDMInfo numa maquina com o toolbox.)
    %
    % CP normal (TS 38.211 5.3.1): 144*Nfft/2048 amostras em todos os
    % simbolos, +16*Nfft/2048 nos simbolos 0 e 7*2^mu de cada subframe.

    scs = carrier.SubcarrierSpacing;
    K = carrier.NSizeGrid * 12;
    mu = log2(scs / 15);

    info.Nfft = max(128, 2^ceil(log2(K / 0.85)));
    info.SampleRate = scs * 1e3 * info.Nfft;
    info.NSubcarriers = K;
    info.SymbolsPerSlot = 14;
    info.SlotsPerSubframe = 2^mu;

    nSymSub = 14 * 2^mu;
    cp = repmat(144 * info.Nfft / 2048, 1, nSymSub);
    longIdx = [1, 1 + 7 * 2^mu];
    cp(longIdx) = cp(longIdx) + 16 * info.Nfft / 2048;

    info.CyclicPrefixLengths = round(cp);                 % por simbolo, 1 subframe
    info.SymbolLengths = info.Nfft + info.CyclicPrefixLengths;
end
