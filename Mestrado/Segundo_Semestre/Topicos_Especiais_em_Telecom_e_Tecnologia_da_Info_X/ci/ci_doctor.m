function ok = ci_doctor()
%CI_DOCTOR Diagnostico do ambiente de simulacao. Nunca falha o CI, so informa.
%
%   A falha mais comum do pipeline e de ambiente, nao de fisica: o pacote 'signal'
%   muda de API conforme a versao do 'statistics' (Octave 5.x usa a API antiga e os
%   scripts .m do signal quebram em runtime, nao em tempo de carregamento). Este
%   script coloca as versoes e as funcoes ausentes no log do job, entao um run
%   vermelho diz em qual categoria o problema esta.

    if exist('OCTAVE_VERSION', 'builtin') ~= 0
        printf('Octave      : %s\n', OCTAVE_VERSION);
    else
        printf('Octave      : ausente (rodando no MATLAB)\n');
        ok = true;  return;
    end

    for p = {'signal', 'statistics', 'communications'}
        v = octave_pkg_version(p{1});
        if any(v), printf('%-12s: %s\n', p{1}, strtrim(sprintf(' %d', v)));
        else,      printf('%-12s: NAO INSTALADO\n', p{1}); end
    end

    try, pkg('load', 'signal'); catch err, printf('pkg load signal -> %s\n', err.message); end
    try, pkg('load', 'statistics'); catch, end

    need = {'resample', 'filtfilt', 'fft', 'ifft'};   % usadas por run_downlink + uwoc_vlc_chain
    found = cellfun(@(n) exist(n, 'file') > 0 || exist(n, 'builtin') > 0 || exist(n, 'var') > 0, need);
    miss  = need(~found);
    if isempty(miss)
        printf('funcoes     : todas as %d resolvem\n', numel(need));
    else
        printf('funcoes     : AUSENTES -> %s\n', strjoin(miss, ', '));
        printf('              se forem do signal, tente o repo var OCTAVE_SIGNAL_API=old\n');
    end

    % Se o signal carrega mas resample nao existe, o break e de API, nao de install.
    if octave_pkg_version('signal')(1) > 0 && exist('resample') == 0
        printf('DIAGNOSTICO : signal instalado porem inerte -- provavel API antiga do statistics.\n');
    end
    ok = true;
end
