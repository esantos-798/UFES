% Salve este arquivo exatamente como: db.m
function y = custom_db(x, varargin)
    % DB Converte magnitude linear para decibéis (dB) de potência
    %   Por padrão, o Toolbox de Comunicações trata a função db() 
    %   como conversão de potência: 10*log10(x)
    
    % Evita logaritmo de zero ou valores negativos que quebram a matemática
    x(x <= 0) = 1e-20; 
    
    % Se o script passar um parâmetro extra como 'voltage', a fórmula teórica 
    % mudaria para 20*log10, mas o padrão do pacote para sinais/potência é 10.
    if nargin > 1 && ischar(varargin{1}) && strcmpi(varargin{1}, 'voltage')
        y = 20 * log10(abs(x));
    else
        % Padrão: conversão de potência
        y = 10 * log10(abs(x));
    end
end