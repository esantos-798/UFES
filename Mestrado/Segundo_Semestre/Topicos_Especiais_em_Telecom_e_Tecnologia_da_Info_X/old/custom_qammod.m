function Y = custom_qammod(X, M)
% MINHA_QAMMOD_NA_UNHA Modulação QAM retangular genérica com código de Gray.
    %   Y = minha_qammod_na_unha(X, M) mapeia os inteiros em X (0 a M-1)
    %   para a constelação M-QAM usando a convenção exata do MATLAB.

    % 1. Validações iniciais
    nbits = log2(M);
    if mod(nbits, 2) ~= 0
        error('M deve ser uma potência par de 2 (ex: 4, 16, 64, 256).');
    end
    if any(X < 0 | X > M-1)
        error('Os elementos de X devem estar no intervalo de 0 a M-1.');
    end

    % Garante o mesmo formato de dimensão da entrada
    Y = zeros(size(X));
    
    % L = número de pontos por linha/coluna no grid (ex: 4 para 16-QAM)
    L = sqrt(M); 
    nbits_por_eixo = nbits / 2;

    % 2. Processa elemento por elemento do vetor X
    for k = 1:numel(X)
        simbolo_int = X(k);
        
        % --- DESENTRELAÇAMENTO DE BITS (Padrão MATLAB/Octave) ---
        % O MATLAB separa os bits do número inteiro alternadamente:
        % Bits ímpares vão para o eixo I, bits pares vão para o eixo Q.
        bits_I = 0;
        bits_Q = 0;
        
        for b = 0:(nbits_por_eixo - 1)
            % Extrai o bit que vai para o eixo I
            bit_atual_I = bitand(bitshift(simbolo_int, -(2*b)), 1);
            bits_I = bitor(bits_I, bitshift(bit_atual_I, b));
            
            % Extrai o bit que vai para o eixo Q
            bit_atual_Q = bitand(bitshift(simbolo_int, -(2*b + 1)), 1);
            bits_Q = bitor(bits_Q, bitshift(bit_atual_Q, b));
        end
        
        % --- DECODIFICAÇÃO DE GRAY INVERSA (Bits para Posição) ---
        % Converte o código de Gray obtido para um índice linear (0 a L-1)
        pos_I = gray2bin_local(bits_I, nbits_por_eixo);
        pos_Q = gray2bin_local(bits_Q, nbits_por_eixo);
        
        % --- CONVERSÃO PARA AMPLITUDE COMPLEXA ---
        % Aplica a fórmula matemática: Amp = 2*pos - (L - 1)
        amp_I = 2 * pos_I - (L - 1);
        amp_Q = 2 * pos_Q - (L - 1);
        
        % O MATLAB inverte o eixo Q verticalmente para a leitura da matriz
        amp_Q = -amp_Q; 
        
        % Monta o número complexo final
        Y(k) = amp_I + 1i * amp_Q;
    end
end

% --- FUNÇÃO AUXILIAR LOCAL ---
% Converte código de Gray de n bits de volta para um inteiro binário comum
function bin = gray2bin_local(gray, nbits)
    bin = 0;
    for i = (nbits-1):-1:0
        bit_gray = bitand(bitshift(gray, -i), 1);
        bit_anterior_bin = bitand(bitshift(bin, -(i+1)), 1);
        bit_bin = bitxor(bit_gray, bit_anterior_bin);
        bin = bitor(bin, bitshift(bit_bin, i));
    end
end