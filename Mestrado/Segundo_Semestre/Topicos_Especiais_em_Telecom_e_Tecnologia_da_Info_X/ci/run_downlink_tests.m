function ok = run_downlink_tests(varargin)
%RUN_DOWNLINK_TESTS Suite de regressao numerica do downlink 5G NR sobre UWOC.
%
%   Roda uma grade de cenarios via run_downlink (backend 'custom', sem toolbox)
%   e compara BER/EVM/SNR contra ci/baseline.csv. Falha se algum numero sair da
%   tolerancia, ou se um cenario lancar excecao.
%
%   Uso local:
%       octave-cli --no-gui -q ci/run_downlink_tests.m
%       matlab -batch "addpath('ci'); run_downlink_tests"
%   Regenere a baseline depois de uma mudanca INTENCIONAL de comportamento:
%       UWOC_SAVE_BASELINE=1 octave-cli --no-gui -q ci/run_downlink_tests.m
%
%   Tolerancia relativa padrao = 2e-3. Octave/MATLAB e versoes distintas de BLAS
%   mudam a mantissa mais baixa; 2e-3 pega regressao real sem gerar falso positivo.

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'ci'));
    addpath(fullfile(root, 'functions_custom'));
    addpath(fullfile(root, 'uwoc'));
    addpath(fullfile(root, 'runner'));

    load_signal_pkg();                       % resample/filtfilt/butter vem daqui; sem ele a suite cai

    ciDir   = fullfile(root, 'ci');
    outDir  = fullfile(ciDir, 'results');
    if ~exist(outDir, 'dir'), mkdir(outDir); end

    relTol  = 2e-3;
    saveBaseline = getenv('UWOC_SAVE_BASELINE') == '1';

    cases   = case_grid();
    metrics = metric_names();

    fprintf('== %d cenarios | backend=custom | tol=%.0e | Octave=%s signal=%s ==\n', ...
        size(cases, 1), relTol, octave_version_str(), pkg_version_str('signal'));

    got = zeros(size(cases, 1), numel(metrics));
    for i = 1:size(cases, 1)
        cfg = cases{i, 1};
        t0 = tic;
        try
            R = run_downlink(cfg);
        catch err
            fprintf('  [FAIL] %-11s %-6s seed=%-3g  excecao: %s\n', cfg.pChannel, ...
                cfg.pDigitalModulation, cfg.seed, err.message);
            ok = false;  return;
        end
        got(i, :) = [R.ber, R.evm_pct, R.evm_noDCnb_pct, R.snr_rx_dB];
        fprintf('  [ok]   %-11s %-6s seed=%-3g  BER=%-8.3g EVM=%7.4f%%  SNR_rx=%8.4f  (%.1fs)\n', ...
            cfg.pChannel, cfg.pDigitalModulation, cfg.seed, R.ber, R.evm_pct, ...
            R.snr_rx_dB, toc(t0));
    end

    baselineFile = fullfile(ciDir, 'baseline.csv');
    if saveBaseline || ~exist(baselineFile, 'file')
        write_table(baselineFile, cases, metrics, got);
        fprintf('\nBaseline gravada em %s (compite este arquivo).\n', baselineFile);
        ok = true;  return;
    end

    [want, missing] = read_table(baselineFile, cases, metrics);
    resultsFile = fullfile(outDir, 'metrics.csv');
    write_table(resultsFile, cases, metrics, got);

    bad = compare(cases, want, got, metrics, relTol);
    if any(missing)
        names = cellfun(@(c) sprintf('%s/%s seed=%g', c{1}.pChannel, c{1}.pDigitalModulation, c{1}.seed), ...
            num2cell(cases(missing, 1), 2), 'UniformOutput', false);
        fprintf('\nFAIL: cenario sem linha na baseline (adicione-o com UWOC_SAVE_BASELINE=1):\n  %s\n', ...
            strjoin(names, '\n  '));
    end
    if isempty(bad) && ~any(missing)
        fprintf('\nPASS: %d cenarios dentro de %.0e de tolerancia.\n', size(cases, 1), relTol);
        fprintf('Detalhes: %s\n', resultsFile);
        write_summary(fullfile(outDir, 'summary.md'), cases, got, metrics, [], relTol);
        ok = true;
    else
        fprintf('\nFAIL: %d valor(es) fora da tolerancia (%.0e):\n', numel(bad), relTol);
        for k = 1:numel(bad)
            b = bad(k);
            fprintf('  %-11s %-6s seed=%-3g  %-14s esperado %-12.6g obtido %-12.6g (diff %.2e)\n', ...
                b.canal, b.mod, b.seed, b.metric, b.wanted, b.got, b.rel);
        end
        fprintf('\nSe a mudanca foi intencional: UWOC_SAVE_BASELINE=1 e commite a nova baseline.\n');
        write_summary(fullfile(outDir, 'summary.md'), cases, got, metrics, bad, relTol);
        ok = false;
    end
end

% -------------------------------------------------------------------------- grade
function cases = case_grid()
    % Coluna 1 = struct de cfg; colunas 2..5 = chaves de indexacao legiveis.
    uwocSnr = struct('SNR_dB', 30, 'SNR_ref', 'rx', 'pdLaw', 'linear', 'I_mod_A', 50);
    mk = @(canal, mod_, seed, uwoc) struct( ...
        'backend', 'custom', 'pChannel', canal, 'pDigitalModulation', mod_, ...
        'seed', seed, 'uwoc', uwoc, 'verbose', false);

    cases = {
        mk('b2b',       'QPSK',  42, struct());
        mk('b2b',       '16QAM', 42, struct());
        mk('uwoc-b2b',  'QPSK',  42, uwocSnr);
        mk('uwoc-b2b',  '16QAM', 42, uwocSnr);
        mk('uwoc-b2b',  '64QAM', 42, uwocSnr);
        mk('uwoc-b2b',  '16QAM',  7, uwocSnr);
        mk('uwoc-b2b',  '16QAM', 42, struct('SNR_dB', 30, 'SNR_ref', 'rx', 'pdLaw', 'square', 'I_mod_A', 50));
        mk('uwoc-b2b',  '16QAM', 42, struct('SNR_dB', 30, 'SNR_ref', 'rx', 'pdLaw', 'linear',  'I_mod_A', 200));
    };
end

function m = metric_names()
    m = {'ber', 'evm_pct', 'evm_noDCnb_pct', 'snr_rx_dB'};
end

function s = octave_version_str()
    if exist('OCTAVE_VERSION', 'builtin') ~= 0, s = ['v' OCTAVE_VERSION]; else, s = 'MATLAB'; end
end

function s = pkg_version_str(name)
    v = octave_pkg_version(name);
    if any(v), s = strtrim(sprintf('%d.', v)); s(end) = []; else, s = 'ausente'; end
end

% -------------------------------------------------------------------------- I/O
function write_table(file, cases, metrics, vals)
    fid = fopen(file, 'w');
    fprintf(fid, 'canal,modulacao,seed,pdLaw,I_mod_A,%s\n', strjoin(metrics, ','));
    for i = 1:size(cases, 1)
        cfg = cases{i, 1};
        fprintf(fid, '%s,%s,%g,%s,%g', cfg.pChannel, cfg.pDigitalModulation, cfg.seed, ...
            opt_field(cfg.uwoc, 'pdLaw', 'n/a'), opt_field(cfg.uwoc, 'I_mod_A', 0));
        for j = 1:numel(metrics)
            % Inf e NaN sao valores legitimos (b2b nao tem ruido), nao baseline ausente: gravar fiel.
            v = vals(i, j);
            if isfinite(v), fprintf(fid, ',%.10g', v); else, fprintf(fid, ',%g', v); end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end

function v = opt_field(s, f, d)
    if isfield(s, f), v = s.(f); else, v = d; end
end

function [want, missing] = read_table(file, cases, metrics)
    fid = fopen(file, 'r');
    if fid < 0, error('read_table:NoFile', 'baseline ausente: %s', file); end
    fgets(fid);                                                    % cabecalho
    rows = textscan(fid, repmat('%s ', 1, 5 + numel(metrics)), 'Delimiter', ',');
    fclose(fid);
    want = NaN(size(cases, 1), numel(metrics));
    for i = 1:size(cases, 1)
        cfg = cases{i, 1};
        key = {cfg.pChannel, cfg.pDigitalModulation, sprintf('%g', cfg.seed)};
        for k = 1:numel(rows{1})
            if ~strcmp(rows{1}{k}, key{1}) || ~strcmp(rows{2}{k}, key{2}) || ~strcmp(rows{3}{k}, key{3})
                continue;
            end
            if ~matches_uwoc(cfg, rows{4}{k}, str2double(rows{5}{k})), continue; end
            for j = 1:numel(metrics)
                want(i, j) = str2double(rows{5 + j}{k});
            end
            break;
        end
    end
    missing = all(isnan(want), 2);                                  % linha nao encontrada na baseline
end

function t = matches_uwoc(cfg, pdLaw, iMod)
    t = strcmp(pdLaw, opt_field(cfg.uwoc, 'pdLaw', 'n/a')) && ...
        abs(iMod - opt_field(cfg.uwoc, 'I_mod_A', 0)) < 1e-9;
end

% -------------------------------------------------------------------------- comparacao
function bad = compare(cases, want, got, metrics, relTol)
    bad = struct('canal', {}, 'mod', {}, 'seed', {}, 'metric', {}, 'wanted', {}, 'got', {}, 'rel', {});
    for i = 1:size(cases, 1)
        cfg = cases{i, 1};
        for j = 1:numel(metrics)
            w = want(i, j);  g = got(i, j);
            if same_value(w, g, relTol), continue; end
            bad(end+1) = struct('canal', cfg.pChannel, 'mod', cfg.pDigitalModulation, ...
                'seed', cfg.seed, 'metric', metrics{j}, 'wanted', w, 'got', g, ...
                'rel', rel_diff(w, g)); %#ok<AGROW>
        end
    end
end

% Inf/NaN comparam-se por igualdade estrutural: o sinal de Inf faz parte do resultado.
function t = same_value(a, b, relTol)
    if isnan(a)
        t = isnan(b);
    elseif isinf(a)
        t = isinf(b) && sign(a) == sign(b);
    else
        t = abs(b - a) <= relTol * max(abs(a), 1e-9);
    end
end

function r = rel_diff(a, b)
    if ~isfinite(a) || ~isfinite(b), r = Inf; else, r = abs(b - a) / max(abs(a), 1e-9); end
end

function write_summary(file, cases, got, metrics, bad, relTol)
    fid = fopen(file, 'w');
    fprintf(fid, '## Downlink 5G NR / UWOC -- resultados\n\n');
    fprintf(fid, '| canal | modulacao | seed | pdLaw | I_mod (mA) |');
    for j = 1:numel(metrics), fprintf(fid, ' %s |', metrics{j}); end
    fprintf(fid, '\n|---|---|---|---|---|');
    for j = 1:numel(metrics), fprintf(fid, '---|'); end
    fprintf(fid, '\n');
    for i = 1:size(cases, 1)
        cfg = cases{i, 1};
        fprintf(fid, '| %s | %s | %g | %s | %g |', cfg.pChannel, cfg.pDigitalModulation, cfg.seed, ...
            opt_field(cfg.uwoc, 'pdLaw', '-'), opt_field(cfg.uwoc, 'I_mod_A', 0));
        for j = 1:numel(metrics)
            if isfinite(got(i, j)), fprintf(fid, ' %.4f |', got(i, j)); else, fprintf(fid, ' - |'); end
        end
        fprintf(fid, '\n');
    end
    if ~isempty(bad)
        fprintf(fid, '\n**%d valor(es) fora da tolerancia %.0e:**\n\n', numel(bad), relTol);
        for k = 1:numel(bad)
            fprintf(fid, '- `%s/%s seed=%g` **%s**: esperado %.6g, obtido %.6g\n', ...
                bad(k).canal, bad(k).mod, bad(k).seed, bad(k).metric, bad(k).wanted, bad(k).got);
        end
    else
        fprintf(fid, '\nTodos os cenarios dentro da tolerancia.\n');
    end
    fclose(fid);
end
