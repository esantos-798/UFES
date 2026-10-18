function carrier = custom_nrCarrierConfig(varargin)
    % custom_nrCarrierConfig Emula nrCarrierConfig (apenas as propriedades usadas)
    %
    % Pares Nome-Valor aceitos: SubcarrierSpacing, NSizeGrid, NSlot,
    % CyclicPrefix ('normal'), NCellID. Nomes desconhecidos geram erro (como
    % no nrCarrierConfig), evitando que um typo passe silenciosamente.

    carrier.SubcarrierSpacing = 15;
    carrier.NSizeGrid = 52;        % default do nrCarrierConfig
    carrier.NSlot = 0;
    carrier.CyclicPrefix = 'normal';
    carrier.NCellID = 1;

    if mod(numel(varargin), 2) ~= 0
        error('custom_nrCarrierConfig:InvalidArgs', 'Argumentos devem ser pares Nome-Valor.');
    end
    for i = 1:2:numel(varargin)
        name = varargin{i};
        if ~ischar(name) || ~isfield(carrier, name)
            error('custom_nrCarrierConfig:InvalidProperty', 'Propriedade invalida: %s', num2str(name));
        end
        carrier.(name) = varargin{i+1};
    end

    if ~any(carrier.SubcarrierSpacing == [15 30 60 120 240 480 960])
        error('custom_nrCarrierConfig:InvalidSCS', 'SubcarrierSpacing invalido.');
    end
    if ~strcmpi(carrier.CyclicPrefix, 'normal')
        error('custom_nrCarrierConfig:CP', 'Apenas CP normal e suportado na emulacao.');
    end
end
