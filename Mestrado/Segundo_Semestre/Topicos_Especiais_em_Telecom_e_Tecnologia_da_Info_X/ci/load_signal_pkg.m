function load_signal_pkg()
%LOAD_SIGNAL_PKG Carrega o pacote 'signal' do Octave de forma segura entre versoes.
%
%   load_signal_pkg()
%
%   run_downlink depende de resample/filtfilt/butter, que so existem com o pacote
%   signal carregado -- logo a suite INTEIRA falha sem ele, e o try/catch silencioso
%   em run_downlink transformaria isso em erro opaco no meio da grade.
%
%   O nome do arquivo de CONFIGURE do signal mudou entre releases (configure.ac ->
%   configure), entao listar o diretorio e mais robusto que adivinhar. Octave sem o
%   pacote (ex.: MATLAB) simplesmente nao faz nada.

    if exist('OCTAVE_VERSION', 'builtin') == 0, return; end   % MATLAB: signal e nativo

    v = octave_pkg_version('signal');
    if ~any(v)                                                % v = [1 4 7]; NaN -> NaN>1.4 = false
        error('load_signal_pkg:Missing', ...
            ['pacote Octave "signal" nao encontrado. No Ubuntu/Debian do CI instale ' ...
             'octave-signal; localmente rode: pkg install -forge signal']);
    end

    ok = try_catch_load();
    if ok, return; end

    d = fullfile(pkg('local_list'), ['signal-' num2str(v(1), '%g') '.' num2str(v(2), '%g') '.' num2str(v(3), '%g')]);
    if any(exist(fullfile(d, 'configure'), 'file'))
        pkg('local_config', 'signal', '--with-octave-Statistics', 'yes');
    end
    if ~try_catch_load()
        error('load_signal_pkg:Load', 'pkg load signal falhou mesmo apos o local_config.');
    end
end

function ok = try_catch_load()
    try
        pkg('load', 'signal');
        ok = exist('resample', 'file') > 0 || exist('resample', 'builtin') > 0;
    catch
        ok = false;
    end
end
