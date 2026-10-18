function d = custom_nrSSS(NID1, NID2)
    % SSS conforme TS 38.211 secao 7.4.2.3 (127 simbolos BPSK, coluna)
    x0 = zeros(1, 127); x1 = zeros(1, 127);
    x0(1:7) = [1 0 0 0 0 0 0];
    x1(1:7) = [1 0 0 0 0 0 0];
    for i = 0:119
        x0(i+8) = mod(x0(i+5) + x0(i+1), 2);
        x1(i+8) = mod(x1(i+2) + x1(i+1), 2);
    end
    m0 = 15*floor(NID1/112) + 5*NID2;
    m1 = mod(NID1, 112);
    n = 0:126;
    d = ((1 - 2*x0(mod(n + m0, 127) + 1)) .* (1 - 2*x1(mod(n + m1, 127) + 1))).';
end
