function d = custom_nrPSS(NID2)
    % PSS conforme TS 38.211 secao 7.4.2.2 (127 simbolos BPSK, coluna)
    x = zeros(1, 127);
    x(1:7) = [0 1 1 0 1 1 1];
    for i = 0:119
        x(i+8) = mod(x(i+5) + x(i+1), 2);
    end
    n = 0:126;
    m = mod(n + 43*NID2, 127);
    d = (1 - 2*x(m+1)).';
end
