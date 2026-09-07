% Salve este arquivo exatamente como: biterr.m
function [number, ratio] = custom_biterr(x, y, varargin)
    % BITERR Calcula o número e a taxa de erro de bit entre dois vetores.
    %   [number, ratio] = biterr(x, y) compara os elementos de x e y.
    
    % 1. Garante que ambos os vetores sejam tratados como vetores coluna longos
    x = x(:);
    y = y(:);
    
    if numel(x) ~= numel(y)
        error('Os vetores de entrada devem ter o mesmo número de elementos.');
    end
    
    % 2. Descobre quantos bits são necessários para representar o maior valor
    max_val = max([max(x), max(y), 1]);
    nbits = ceil(log2(double(max_val + 1)));
    
    % Se o seu script estiver passando vetores que já são binários (0 e 1), 
    % nbits será 1 e a conversão dec2bin funcionará perfeitamente.
    
    % 3. Converte os números decimais para matrizes de caracteres binários ('0' e '1')
    bin_x = dec2bin(x, nbits);
    bin_y = dec2bin(y, nbits);
    
    % 4. Compara caractere por caractere (onde for diferente, vira 1)
    erros_matriz = (bin_x ~= bin_y);
    
    % 5. Soma todos os bits errados
    number = sum(erros_matriz(:));
    
    % 6. Calcula a taxa de erro (BER) dividindo pelo total absoluto de bits
    total_bits = numel(bin_x);
    ratio = number / total_bits;
end