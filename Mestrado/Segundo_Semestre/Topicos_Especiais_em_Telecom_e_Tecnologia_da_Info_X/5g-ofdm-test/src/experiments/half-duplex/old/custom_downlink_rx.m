function [rxCellid, pdschRxSym, pdschEqSym, hest, decbits, ber, EVM_porc] = ...
    custom_downlink_rx(rxGrid, DLgrid, pModulationLevel, trBlk, pNCellID)

    rxCellid = pNCellID;
    pdschRxSym = rxGrid(:);
    txSymbols = DLgrid(:);
    
    % --- EQUALIZAÇÃO SIMPLES ZERO-FORCING (ZF) ---
    % Estima o fator de canal complexo médio (fase e magnitude)
    H_est = mean(pdschRxSym ./ (txSymbols + eps));
    hest = H_est * ones(size(rxGrid));
    
    % Aplica a equalização compensando o canal e fase residual
    pdschEqSym = pdschRxSym / H_est;
    
    % Demodulação e mapeamento de bits
    symbols_int_rx = qamdemod(pdschEqSym, pModulationLevel);
    k = log2(pModulationLevel);
    bits_matrix_rx = de2bi(symbols_int_rx, k, 'left-msb');
    rxBits = reshape(bits_matrix_rx.', [], 1);
    
    decbits = rxBits(1:length(trBlk));
    [num_erros, ber] = biterr(trBlk, decbits);
    fprintf('BER = %.4f, Errors = %d\n', ber, num_erros);

    [~, EVM_porc] = custom_evm(pModulationLevel, txSymbols, pdschEqSym);
    fprintf('EVM Percentual = %.2f%%\n', EVM_porc);
end