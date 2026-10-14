function [rxCellid, pdschRxSym, pdschEqSym, hest, decbits, ber, EVM_porc] = ...
    custom_downlink_rx(rxGrid, pSizeGrid, pModulationLevel, trBlk, pNCellID, txInfo)
    % custom_downlink_rx Emula o RX do main_downlink.m SEM 5G Toolbox.
    %
    % Fluxo espelhando o original: detectPSS -> detecSSS -> estimacao de canal
    % por DMRS -> extracao dos RE do PDSCH -> equalizador MMSE -> demodulacao
    % -> BER e EVM. Sem decodificacao LDPC: BER e pre-FEC.

    % Guarda: pModulationLevel precisa corresponder a pDigitalModulation usada no TX.
    % Sem isso a demodulacao usa outra constelacao e o BER fica ~0.5 com EVM baixo.
    if pModulationLevel ~= txInfo.M
        error('custom_downlink_rx:ModLevel', ...
            ['pModulationLevel (%d) difere da modulacao transmitida (M = %d). ', ...
             'Ajuste pDigitalModulation e pModulationLevel juntos (QPSK=4, 16QAM=16, 64QAM=64).'], ...
            pModulationLevel, txInfo.M);
    end

    % Deteccao de celula (cega)
    NID2 = custom_detectPSS(rxGrid, pSizeGrid);
    pbCHoffset = round((pSizeGrid*12 - 240)/2);
    startSSS = pbCHoffset + 57;
    sssRx = rxGrid(startSSS:startSSS+126, 4);
    [NID1, rxCellid] = custom_detecSSS(sssRx, NID2, true);
    if rxCellid ~= pNCellID
        warning('custom_downlink_rx:CellID', 'Cell ID detectado (%d) difere do transmitido (%d).', rxCellid, pNCellID);
    end

    % PDSCH apos o burst SS (slot 2)
    rxGridPDSCH = rxGrid(:, 15:end);

    % Estimacao de canal via DMRS
    [hest, nVar] = custom_nrChannelEstimate(rxGridPDSCH, txInfo.dmrsIndices, txInfo.dmrsSymbols);

    % Extracao e equalizacao MMSE
    pdschRxSym = rxGridPDSCH(txInfo.pdschIndices);
    pdschHest = hest(txInfo.pdschIndices);
    pdschEqSym = conj(pdschHest) .* pdschRxSym ./ (abs(pdschHest).^2 + nVar);

    % Demodulacao (decisao rigida) e BER
    decbits = custom_nrSymbolDemodulate(pdschEqSym, pModulationLevel);
    n = min(numel(trBlk), numel(decbits));
    num_erros = sum(trBlk(1:n) ~= decbits(1:n)) + (numel(trBlk) - n);
    ber = num_erros / numel(trBlk);
    output = sprintf('BER = %.4f, Errors = %d', ber, num_erros);
    disp(output);

    % EVM
    [~, EVM_porc] = custom_evm(pModulationLevel, txInfo.pdschSymbols, pdschEqSym);
    fprintf('EVM Percentual = %.2f', EVM_porc);
end
