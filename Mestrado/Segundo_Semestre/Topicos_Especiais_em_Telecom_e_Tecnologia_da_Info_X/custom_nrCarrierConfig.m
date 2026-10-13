function carrier = custom_nrCarrierConfig(varargin)
    % custom_nrCarrierConfig Emula a criação da estrutura de portadora 5G NR
    %
    % Aceita pares de Nome-Valor, como:
    % 'SubcarrierSpacing', 15
    % 'NSizeGrid', 25

    % Valores padrão do 5G NR (SCS 15kHz, 25 RBs)
    carrier.SubcarrierSpacing = 15;
    carrier.NSizeGrid = 25;

    % Parseia os argumentos de entrada
    for i = 1:2:length(varargin)
        if strcmp(varargin{i}, 'SubcarrierSpacing')
            carrier.SubcarrierSpacing = varargin{i+1};
        elseif strcmp(varargin{i}, 'NSizeGrid')
            carrier.NSizeGrid = varargin{i+1};
        end
    end
end
