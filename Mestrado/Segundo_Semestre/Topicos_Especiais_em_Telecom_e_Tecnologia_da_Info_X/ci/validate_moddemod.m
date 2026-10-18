function ok = validate_moddemod()
%VALIDATE_MODEMODMOD custom_modulate/custom_demod contra modulate/demod reais.
%
%   ok = validate_moddemod()
%
%   Versao headless com ASSERCES de octave_5g/valida_custom_modulate_demod.m, que
%   apenas imprimia numeros (um humano precisava ler e julgar). aqui o job falha
%   sozinho se o DUC/DDC divergir.
%
%   So roda quando modulate/demod reais existem. Em Octave sem o pacote signal eles
%   nao existem, entao o job reporta SKIP: comparar custom_modulate contra si mesma
%   (ambos em octave_5g/, conforme a ordem do path) seria um falso verde.

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root, 'functions_custom'));   % unicos custom_* no path: evita sombra

    if exist('modulate', 'file') == 0 || exist('demod', 'file') == 0
        printf('SKIP: modulate/demod reais indisponiveis (rode no MATLAB com Signal Processing Toolbox)\n');
        ok = true;  return;
    end

    fs = 75e6;  fc = 5e6;  N = 20000;
    rng(1);
    bb = filter(ones(1, 8)/8, 1, randn(N, 1) + 1i*randn(N, 1));
    [b, a] = butter(6, 2.25e6/(fs/2));
    bb = filtfilt(b, a, real(bb)) + 1i*filtfilt(b, a, imag(bb));

    y_ref = modulate(real(bb), fc, fs, 'qam', imag(bb));
    y_cus = custom_modulate(real(bb), fc, fs, 'qam', imag(bb));
    duc = max(abs(y_ref - y_cus));
    printf('DUC : max |modulate - custom_modulate| = %.3e (esperado ~1e-15)\n', duc);

    [ri, rq] = demod(y_ref, fc, fs, 'qam');
    [ci, cq] = custom_demod(y_ref, fc, fs, 'qam');
    r = ri(:) + 1i*rq(:);  c = ci(:) + 1i*cq(:);
    g = (c' * r) / (c' * c);
    ddc = norm(r - g*c) / norm(r);
    printf('DDC : razao demod/custom = %.4f   erro apos remover a razao = %.3e (esperado ~1e-12)\n', real(g), ddc);

    ok = duc < 1e-10 && ddc < 1e-8;
    if ok, printf('PASS: DUC/DDC equivalentes ao referencial.\n');
    else,   printf('FAIL: divergencia acima da tolerancia.\n'); end
end
