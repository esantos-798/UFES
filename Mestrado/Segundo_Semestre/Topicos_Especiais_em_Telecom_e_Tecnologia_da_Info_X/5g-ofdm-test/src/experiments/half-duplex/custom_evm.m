function [EVM_dB, EVM_porc] = custom_evm(mLevel, txSym, rxSym)
    % custom_evm Replica o evm.m do laboratorio (Jair Silva, 2009).
    %
    % Tx e Rx sao normalizados SEPARADAMENTE para potencia media 1 (modnorm 'avpow',1), o que
    % remove diferencas de ganho (por ex. o vies do MMSE). A constelacao de referencia
    % normalizada tem potencia media 1, entao o denominador e 1:
    %   EVMrms = sqrt( mean|tx_n - rx_n|^2 );  EVM_dB = 20 log10(EVMrms);  EVM_porc = 100*EVMrms
    % mLevel e aceito por compatibilidade (nao altera o resultado).

    txSym = txSym(:);
    rxSym = rxSym(:);
    minLen = min(length(txSym), length(rxSym));
    txSym = txSym(1:minLen);
    rxSym = rxSym(1:minLen);

    txn = txSym / sqrt(mean(abs(txSym).^2));
    rxn = rxSym / sqrt(mean(abs(rxSym).^2));

    EVMrms   = sqrt(mean(abs(txn - rxn).^2));
    EVM_dB   = 20*log10(EVMrms);
    EVM_porc = 100*10^(EVM_dB/20);
end
