% valida_custom_modulate_demod.m  -- RODAR NO MATLAB (precisa do Signal Processing Toolbox)
% Compara custom_modulate/custom_demod com modulate/demod reais.
% Esperado: erro do DUC ~ 1e-16 (mesma formula); DDC: razao de ganho constante
% (1 se o demod real nao tem fator extra, 2 ou 0.5 caso contrario) e erro residual ~ 0
% depois de remover essa razao. O main normaliza o RX, entao o ganho nao afeta o resultado.

if exist('modulate','file') == 0 || exist('demod','file') == 0
    error('modulate/demod reais nao encontrados: rode este teste no MATLAB com Signal Processing Toolbox.');
end

fs = 75e6;  fc = 5e6;  N = 20000;
rng(1);
bb = filter(ones(1,8)/8, 1, randn(N,1) + 1i*randn(N,1));
[b,a] = butter(6, 2.25e6/(fs/2));
bb = filtfilt(b,a,real(bb)) + 1i*filtfilt(b,a,imag(bb));

% --- DUC ---
y_ref = modulate(real(bb), fc, fs, 'qam', imag(bb));
y_cus = custom_modulate(real(bb), fc, fs, 'qam', imag(bb));
fprintf('DUC : erro max |modulate - custom_modulate| = %.2e (esperado ~1e-15)\n', max(abs(y_ref - y_cus)));

% --- DDC (sobre o MESMO sinal passa-faixa) ---
[ri, rq] = demod(y_ref, fc, fs, 'qam');
[ci, cq] = custom_demod(y_ref, fc, fs, 'qam');
r = ri + 1i*rq;  c = ci + 1i*cq;
g = (c'*r) / (c'*c);
fprintf('DDC : razao demod_real / custom_demod = %.4f  (1 = ganho identico)\n', real(g));
fprintf('DDC : erro relativo apos remover a razao   = %.2e (esperado ~1e-12)\n', norm(r - g*c)/norm(r));

% --- orientacao (linha -> linha), como o main espera ---
yr = custom_modulate(real(bb).', fc, fs, 'qam', imag(bb).');
[xr, ~] = custom_demod(yr, fc, fs, 'qam');
fprintf('Orientacao: demod(linha) -> %s (esperado 1x%d)\n', mat2str(size(xr)), N);
