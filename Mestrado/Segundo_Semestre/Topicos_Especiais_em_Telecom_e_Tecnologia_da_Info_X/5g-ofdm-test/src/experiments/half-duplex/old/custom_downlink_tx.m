function [DLgrid, waveform, cInfo, pdschSymbols, trBlk] = custom_downlink_tx(cCarrier, pSizeGrid, pDigitalModulation)
    % Dimensões padrão para 1 slot
    N_subcarriers = 12 * pSizeGrid; % 300 subportadoras
    N_symbols = 14; 
    
    % Mapeamento do nível de modulação
    switch upper(pDigitalModulation)
        case 'QPSK'
            M = 4;
        case '16QAM'
            M = 16;
        case '64QAM'
            M = 64;
        otherwise
            M = 4;
    end
    
    k = log2(M); % Bits por símbolo
    numBits = N_subcarriers * N_symbols * k;
    trBlk = randi([0 1], numBits, 1);
    
    % --- CONVERSÃO DE BITS PARA SÍMBOLOS INTEIROS (COMPATÍVEL COM OCTAVE) ---
    bits_matrix = reshape(trBlk, k, []).';
    symbols_int = bi2de(bits_matrix, 'left-msb');
    
    % Modulação QAM no Octave
    symbols = qammod(symbols_int, M);
    
    % Normalização da potência média para 1
    symbols = symbols / sqrt(mean(abs(symbols).^2));
    
    % Montagem da grade
    DLgrid = reshape(symbols, N_subcarriers, N_symbols);
    pdschSymbols = DLgrid(:);
    
    % Modulação OFDM customizada
    [waveform, cInfo] = custom_nrOFDMModulate(cCarrier, DLgrid);
end