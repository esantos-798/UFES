function BER_teorica = calcular_ber_teorica_fallback(OSNR_dB_vetor, M)
    % CALCULAR_BER_TEORICA_FALLBACK Calcula a BER teórica para M-QAM (AWGN)
    % sem depender do Communications Toolbox.
    
    k_bits = log2(M); % Bits por símbolo (ex: 2 para 4-QAM, 4 para 16-QAM)
    
    % Converte OSNR (Es/N0) de dB para linear
    OSNR_linear = 10.^(OSNR_dB_vetor / 10);
    
    % Converte Es/N0 para Eb/N0
    EbN0_linear = OSNR_linear / k_bits;
    
    if M == 4
        % Fórmula exata para QPSK / 4-QAM
        BER_teorica = 0.5 * erfc(sqrt(EbN0_linear));
        
    elseif M == 16
        % Fórmula exata para 16-QAM
        termo_q = sqrt((3 * k_bits * EbN0_linear) / (2 * (M - 1)));
        BER_teorica = (2 / k_bits) * (1 - 1 / sqrt(M)) * erfc(termo_q);
        
    else
        % Fallback analítico aproximado para outros valores de M-QAM
        termo_q = sqrt((3 * k_bits * EbN0_linear) / (2 * (M - 1)));
        BER_teorica = (4 / k_bits) * (1 - 1 / sqrt(M)) * (0.5 * erfc(termo_q / sqrt(2)));
    end
end