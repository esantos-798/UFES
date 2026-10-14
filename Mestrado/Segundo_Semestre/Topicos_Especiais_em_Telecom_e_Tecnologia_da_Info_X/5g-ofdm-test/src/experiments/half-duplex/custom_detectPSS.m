function NID2 = custom_detectPSS(rxGrid, pSizeGrid)
    % custom_detectPSS Detecta NID2 correlando o PSS recebido (simbolo 2 da grade)
    % com as 3 sequencias possiveis. Layout do SSB conforme custom_downlink_tx.
    pbCHoffset = round((pSizeGrid*12 - 240)/2);
    rxPSS = rxGrid(pbCHoffset+57 : pbCHoffset+183, 2);
    metric = zeros(1, 3);
    for n = 0:2
        metric(n+1) = abs(sum(rxPSS(:) .* conj(custom_nrPSS(n))));
    end
    [~, i] = max(metric);
    NID2 = i - 1;
end
