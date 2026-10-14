function sym = custom_nrSymbolModulate(bits, M)
    % Mapeamento QPSK/16QAM/64QAM conforme TS 38.211 secao 5.1 (Gray, potencia media unitaria).
    % bits: vetor de bits (comprimento multiplo de log2(M)). Sem dependencia de qammod/bi2de.
    k = log2(M);
    b = reshape(double(bits(:)), k, []).';
    s = 1 - 2*b;
    switch M
        case 4
            sym = (s(:,1) + 1i*s(:,2)) / sqrt(2);
        case 16
            sym = (s(:,1).*(2 - s(:,3)) + 1i*s(:,2).*(2 - s(:,4))) / sqrt(10);
        case 64
            sym = (s(:,1).*(4 - s(:,3).*(2 - s(:,5))) + ...
                   1i*s(:,2).*(4 - s(:,4).*(2 - s(:,6)))) / sqrt(42);
        otherwise
            error('custom_nrSymbolModulate:M', 'Modulacao nao suportada: M = %d', M);
    end
end
