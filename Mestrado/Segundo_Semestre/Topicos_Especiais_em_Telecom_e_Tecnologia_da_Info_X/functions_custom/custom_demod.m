function [x1, x2] = custom_demod(y, fc, fs, method)
    % custom_demod Substituto do demod() do Signal Processing Toolbox (apenas 'qam')
    %
    %   [I, Q] = custom_demod(y, fc, fs, 'qam')
    %
    % Replica o demod.m do Signal Processing Toolbox (conferido contra o codigo-fonte, R2024):
    %   x1 = filtfilt(b,a, 2*y.*cos(2*pi*fc*t));  x2 = filtfilt(b,a, 2*y.*sin(2*pi*fc*t));
    %   [b,a] = butter(5, fc*2/fs)
    % O fator 2 devolve I e Q com a amplitude original (ida e volta com custom_modulate = identidade).
    % Orientacao preservada (linha -> linha, coluna -> coluna).

    if nargin < 4 || ~ischar(method) || ~strcmpi(method, 'qam')
        error('custom_demod:Method', 'Apenas o metodo ''qam'' e suportado.');
    end
    if fc*2/fs >= 1
        error('custom_demod:Fc', 'fc deve ser menor que fs/2.');
    end

    wasRow = isrow(y);
    if wasRow
        y = y.';
    end
    t = (0:size(y, 1)-1).' / fs;
    [b, a] = butter(5, fc*2/fs);
    x1 = filtfilt(b, a, 2 * y .* cos(2*pi*fc*t));
    x2 = filtfilt(b, a, 2 * y .* sin(2*pi*fc*t));
    if wasRow
        x1 = x1.'; x2 = x2.';
    end
end
