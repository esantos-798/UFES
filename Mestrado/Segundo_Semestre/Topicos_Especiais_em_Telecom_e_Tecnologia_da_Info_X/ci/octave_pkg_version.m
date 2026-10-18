function verNum = octave_pkg_version(pkgName)
%OCTAVE_PKG_VERSION Versao instalada de um pacote do Octave, ou NaN se ausente.
%
%   verNum = octave_pkg_version('signal')     % signal-1.4.7  ->  [1 4 7]
%
%   O nome do argumento NAO pode ser 'pkg': isso sombrearia o builtin pkg() e a
%   chamada pkg('list') viraria "indexar um string". Octave >= 6 guarda a versao
%   como numero e Octave 4/5 como string; pkg('list') retorna cell de structs
%   (Octave 11) ou struct array (versoes anteriores). Aceitar os dois formatos e
%   o que faz este helper funcionar no runner do GitHub (Ubuntu) e no Windows.

    verNum = NaN;
    try
        c = pkg('list');
        if iscell(c)
            idx = find(strcmp(pkgName, cellfun(@(e) e.name, c, 'UniformOutput', false)), 1);
            if isempty(idx), return; end
            d = c{idx}.version;
        else
            idx = find(strcmp(pkgName, {c.name}), 1);
            if isempty(idx), return; end
            d = c(idx).version;
        end
        if iscell(d), if isempty(d), return; end, d = d{1}; end
        if isnumeric(d)
            verNum = d(1:min(3, numel(d)));
        elseif ischar(d)
            t = sscanf(strtrim(d), '%d.%d.%d');
            if ~isempty(t), verNum = t(1:min(3, numel(t))); end
        end
    catch
        verNum = NaN;
    end
end
