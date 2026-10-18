function bits = custom_nrSymbolDemodulate(sym, M)
    % Decisao rigida (minima distancia) para o mapeamento de custom_nrSymbolModulate.
    % Espera simbolos na MESMA escala do transmissor (potencia media unitaria).
    k = log2(M);
    idx = (0:M-1).';
    tableBits = zeros(M, k);
    for j = 1:k
        tableBits(:, j) = mod(floor(idx / 2^(k-j)), 2);
    end
    tableSym = custom_nrSymbolModulate(reshape(tableBits.', [], 1), M);

    sym = sym(:);
    D = abs(sym - tableSym.');          % N-por-M
    [~, ix] = min(D, [], 2);
    bits = reshape(tableBits(ix, :).', [], 1);
end
