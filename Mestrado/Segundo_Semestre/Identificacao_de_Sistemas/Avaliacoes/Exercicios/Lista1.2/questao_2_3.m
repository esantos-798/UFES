%% ========================================================================
%  Questao 2.3 - Qual a ordem e os parametros do modelo ARMA que
%  representa a serie temporal y(k)?
%
%  Metodo: analisar o comportamento da ACF (r(k), ja calculada na 2.1) e
%  da PACF (phi_kk), usando a DUALIDADE descrita tanto nas notas do
%  professor quanto no livro da Profa. Monica Barros (Exemplos 5.4.23 a
%  5.4.27):
%
%      AR(p):  ACF  decai (exponenciais/senoides amortecidas)
%              PACF tem CORTE BRUSCO no lag p
%
%      MA(q):  ACF  tem CORTE BRUSCO no lag q
%              PACF decai (exponenciais/senoides amortecidas)
%
%      ARMA(p,q): ambas decaem (sem corte brusco em nenhuma das duas)
%
%  A PACF e calculada resolvendo as equacoes de Yule-Walker (sistema
%  5.4.14 do livro / sistema analogo nas notas do professor) via
%  regra de Cramer, SEM usar parcorr()/aryule() (evita Statistics/
%  Econometrics Toolbox e o pacote 'signal' do Octave).
%
%  Compativel com MATLAB e GNU Octave.
% ========================================================================
clear all; close all; clc;

%% --- 0) Reaproveita y(k) e r(k) da 2.1 (ou recalcula se necessario) -------
I = 5;
if exist('resultado_questao2_1.mat', 'file')
    load('resultado_questao2_1.mat', 'y', 'T', 'r', 'Kmax');
else
    [y] = estocastico(I); y = y(:); T = numel(y);
    Kmax = 50;
    Zbar = mean(y);
    denominador = sum((y-Zbar).^2);
    r = zeros(Kmax+1,1);
    for k = 0:Kmax
        idx = 1:(T-k);
        r(k+1) = sum((y(idx)-Zbar).*(y(idx+k)-Zbar)) / denominador;
    end
end
limite = 2/sqrt(T);

%% --- 1) PACF via equacoes de Yule-Walker (regra de Cramer) -----------------
Pmax = 20;   % numero maximo de lags da PACF a calcular
phi_kk = zeros(Pmax, 1);

for k = 1:Pmax
    % Monta a matriz de autocorrelacoes R (k x k), Toeplitz, e o vetor rhs
    R = zeros(k, k);
    for i = 1:k
        for j = 1:k
            R(i,j) = r(abs(i-j) + 1);   % r(+1) porque r(1) = r(lag 0)
        end
    end
    rhs = r(2:k+1);              % [r(1), r(2), ..., r(k)]'

    % Regra de Cramer: phi_kk = det(R com ultima coluna = rhs) / det(R)
    R_num = R;
    R_num(:, end) = rhs;
    phi_kk(k) = det(R_num) / det(R);
end

%% --- 2) Limite de significancia da PACF (mesma formula de Bartlett) -------
limite_pacf = 2/sqrt(T);

%% --- 3) Graficos lado a lado: ACF e PACF -----------------------------------
figure('Name', 'Questao 2.3 - ACF e PACF para identificacao do modelo');

subplot(1,2,1);
lags_acf = 0:Kmax;
stem(lags_acf, r, 'filled', 'MarkerSize', 4);
hold on;
plot(lags_acf, limite*ones(size(lags_acf)), 'r--');
plot(lags_acf, -limite*ones(size(lags_acf)), 'r--');
hold off;
grid on; xlabel('Lag k'); ylabel('r(k)'); title('ACF');

subplot(1,2,2);
lags_pacf = 1:Pmax;
stem(lags_pacf, phi_kk, 'filled', 'MarkerSize', 4);
hold on;
plot(lags_pacf, limite_pacf*ones(size(lags_pacf)), 'r--');
plot(lags_pacf, -limite_pacf*ones(size(lags_pacf)), 'r--');
hold off;
grid on; xlabel('Lag k'); ylabel('\phi_{kk}', 'Interpreter', 'none'); title('PACF');

drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'acf_pacf_2_3.png');
fprintf('Figura salva: acf_pacf_2_3.png\n');

%% --- 4) Lags significativos de cada funcao ---------------------------------
sig_acf  = find(abs(r(2:end)) > limite);        % ignora lag 0 (sempre 1)
sig_pacf = find(abs(phi_kk) > limite_pacf);

fprintf('\nLags da ACF estatisticamente significativos : '); disp(sig_acf');
fprintf('Lags da PACF estatisticamente significativos: '); disp(sig_pacf');

fprintf(['\n--- Como interpretar (olhando o grafico/listas acima) ---\n' ...
         '- Se a PACF tiver corte brusco no lag p (e a ACF decair aos poucos)\n' ...
         '  -> modelo AR(p).\n' ...
         '- Se a ACF tiver corte brusco no lag q (e a PACF decair aos poucos)\n' ...
         '  -> modelo MA(q).\n' ...
         '- Se nenhuma das duas tiver corte brusco (ambas decaem)\n' ...
         '  -> modelo ARMA(p,q) misto (vide exercicio 2.4, que ja fornece a\n' ...
         '  estrutura ARMA(3,2) teorica do processo gerador).\n']);