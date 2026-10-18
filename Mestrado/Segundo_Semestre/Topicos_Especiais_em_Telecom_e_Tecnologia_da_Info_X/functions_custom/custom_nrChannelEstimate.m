function [hest, nVar] = custom_nrChannelEstimate(rxGrid, dmrsIndices, dmrsSymbols)
    % custom_nrChannelEstimate Estimativa de canal por DMRS (analoga ao nrChannelEstimate)
    %
    % LS nos RE de DMRS -> suavizacao em frequencia (janela de 3 pilotos) ->
    % interpolacao linear em frequencia e tempo. nVar e estimada a partir
    % do residuo LS - suavizado. NAO e bit-exata com o nrChannelEstimate.

    [K, L] = size(rxGrid);
    hLS = rxGrid(dmrsIndices(:)) ./ dmrsSymbols(:);
    [sc, sy] = ind2sub([K L], dmrsIndices(:));
    symList = unique(sy).';

    W = 3;
    ker = ones(W, 1);
    Hsym = zeros(K, numel(symList));
    nAcc = 0; nCnt = 0;
    for i = 1:numel(symList)
        sel = (sy == symList(i));
        [scs, ord] = sort(sc(sel));
        h = hLS(sel); h = h(ord);

        cnt = conv(ones(size(h)), ker, 'same');
        hs = conv(h, ker, 'same') ./ cnt;

        nAcc = nAcc + sum(abs(h - hs).^2 ./ (1 - 1./cnt));
        nCnt = nCnt + numel(h);

        q = (1:K).';
        Hsym(:, i) = interp1(scs, real(hs), q, 'linear', 'extrap') + ...
                1i*interp1(scs, imag(hs), q, 'linear', 'extrap');
    end
    nVar = nAcc / nCnt;

    if numel(symList) == 1
        hest = repmat(Hsym, 1, L);
    else
        t = 1:L;
        hest = (interp1(symList, real(Hsym).', t, 'linear', 'extrap') + ...
            1i*interp1(symList, imag(Hsym).', t, 'linear', 'extrap')).';
    end
end
