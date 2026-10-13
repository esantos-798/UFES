% Salve este arquivo exatamente como: qamdemod.m
function X = custom_qamdemod(Y, M)
    % QAMDEMOD Demodulação QAM retangular genérica com código de Gray.
    %   X = qamdemod(Y, M) recebe o vetor complexo Y e recupera os
    %   valores inteiros (0 a M-1) correspondentes à constelação padrão.

    % Inicializa o vetor de saída com as mesmas dimensões de Y
    X = zeros(size(Y));
    
    % Para encontrar o vizinho mais próximo de forma genérica para qualquer M,
    % nós primeiro geramos a constelação ideal completa usando a função que você já criou.
    constelacao_ideal = custom_qammod(0:(M-1), M);
    
    % Processa cada símbolo recebido
    for k = 1:numel(Y)
        ponto_recebido = Y(k);
        
        % Calcula a distância euclidiana quadrada entre o ponto recebido 
        % e TODOS os M pontos possíveis da constelação ideal
        distancias = abs(ponto_recebido - constelacao_ideal).^2;
        
        % Encontra o índice do ponto que tem a menor distância (mínimo)
        [~, indice_mais_proximo] = min(distancias);
        
        % Como os índices no Octave começam em 1, e nossos símbolos vão de 0 a M-1,
        % o símbolo estimado será o índice - 1.
        X(k) = indice_mais_proximo - 1;
    end
end