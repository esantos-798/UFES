function tbs = custom_nrTBS(modulation, nLayers, nPRB, NREPerPRB, codeRate, xOh)
    % custom_nrTBS Tamanho do bloco de transporte do PDSCH (TS 38.214, secao 5.1.3.2), como o nrTBS
    % usado em initDLSCH.m:  trBlkSizes = nrTBS(mod, nLayers, nPRB, NREPerPRB, codeRate, Xoh)
    % Informativo: a cadeia custom nao usa LDPC. [a confirmar] contra o nrTBS do toolbox.

    switch upper(modulation)
        case 'QPSK',   Qm = 2;
        case '16QAM',  Qm = 4;
        case '64QAM',  Qm = 6;
        case '256QAM', Qm = 8;
        otherwise, error('custom_nrTBS:Mod', 'Modulacao nao suportada: %s', modulation);
    end
    Nre = min(156, NREPerPRB - xOh) * nPRB;
    Ninfo = Nre * codeRate * Qm * nLayers;

    if Ninfo <= 3824
        n = max(3, floor(log2(Ninfo)) - 6);
        Ninfo1 = max(24, 2^n * floor(Ninfo / 2^n));
        T = [24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 208 224 ...
             240 256 272 288 304 320 336 352 368 384 408 432 456 480 504 528 552 576 608 640 672 704 ...
             736 768 808 848 888 928 984 1032 1064 1128 1160 1192 1224 1256 1288 1320 1352 1416 1480 ...
             1544 1608 1672 1736 1800 1864 1928 2024 2088 2152 2216 2280 2408 2472 2536 2600 2664 ...
             2728 2792 2856 2976 3104 3240 3368 3496 3624 3752 3824];
        tbs = min(T(T >= Ninfo1));
    else
        n = floor(log2(Ninfo - 24)) - 5;
        Ninfo1 = max(3840, 2^n * round((Ninfo - 24) / 2^n));
        if codeRate <= 1/4
            C = ceil((Ninfo1 + 24) / 3816);
            tbs = 8*C*ceil((Ninfo1 + 24) / (8*C)) - 24;
        elseif Ninfo1 > 8424
            C = ceil((Ninfo1 + 24) / 8424);
            tbs = 8*C*ceil((Ninfo1 + 24) / (8*C)) - 24;
        else
            tbs = 8*ceil((Ninfo1 + 24) / 8) - 24;
        end
    end
end
