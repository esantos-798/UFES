function [y, t] = custom_modulate(x, fc, fs, method, q)
    % custom_modulate Substituto do modulate() do Signal Processing Toolbox (apenas 'qam')
    %
    %   y = custom_modulate(I, fc, fs, 'qam', Q)
    %
    % Replica o modulate.m do Signal Processing Toolbox (conferido contra o codigo-fonte, R2024):
    %   y = I.*cos(2*pi*fc*t) + Q.*sin(2*pi*fc*t)
    % I e Q reais, mesmo tamanho. Vetor linha entra -> linha sai; vetor coluna -> coluna
    % (necessario porque o main usa .' depois do DUC). Matrizes: processa por colunas.
    % Obs.: y = Re{(I - jQ) e^{j 2 pi fc t}}  (convencao do MATLAB, nao a de (I+jQ)).

    if nargin < 5 || ~ischar(method) || ~strcmpi(method, 'qam')
        error('custom_modulate:Method', 'Apenas o metodo ''qam'' com Q como 5o argumento e suportado.');
    end
    if ~isreal(x) || ~isreal(q)
        error('custom_modulate:Real', 'I e Q devem ser reais.');
    end
    if ~isequal(size(x), size(q))
        error('custom_modulate:Size', 'I e Q devem ter o mesmo tamanho.');
    end

    wasRow = isrow(x);
    if wasRow
        x = x.'; q = q.';
    end
    t = (0:size(x, 1)-1).' / fs;
    y = x .* cos(2*pi*fc*t) + q .* sin(2*pi*fc*t);
    if wasRow
        y = y.'; t = t.';
    end
end
