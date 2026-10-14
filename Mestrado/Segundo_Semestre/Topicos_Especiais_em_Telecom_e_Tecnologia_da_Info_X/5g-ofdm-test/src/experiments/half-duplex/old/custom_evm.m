function [EVM_dB, EVM_porc] = custom_evm(mLevel, txSym, rxSym)
    % custom_evm Calcula o EVM (Error Vector Magnitude) entre símbolos Tx e Rx
    
    txSym = txSym(:);
    rxSym = rxSym(:);
    
    % Ajusta comprimentos se necessário
    minLen = min(length(txSym), length(rxSym));
    txSym = txSym(1:minLen);
    rxSym = rxSym(1:minLen);
    
    % Vetor de erro
    errorVector = rxSym - txSym;
    
    % EVM RMS
    evm_rms = sqrt(mean(abs(errorVector).^2) / mean(abs(txSym).^2));
    
    EVM_porc = evm_rms * 100;
    EVM_dB = 20 * log10(evm_rms + eps);
end