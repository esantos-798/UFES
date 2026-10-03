%% ========================================================================
%  Questao 1.4 - Numero minimo de municipios (amostras) para estimar a
%  media de emissao de CO2 com 98% de confianca e erro < 10.000 tCO2/ano
%
%  ATENCAO - premissa assumida: o enunciado nao diz de qual ANO usar o
%  desvio padrao. Como a questao 1.5 (logo em seguida) fala explicitamente
%  de 2024 e reaproveita os valores da 1.1, assumimos aqui tambem o ano de
%  2024. Se o professor pediu outro ano, so trocar a variavel ANO_ALVO
%  abaixo (o script calcula para todos os anos, para referencia).
%
%  Formula classica (populacao "infinita"):
%       n0 = (z_(alpha/2) * sigma / E)^2
%
%  Como aqui a populacao e FINITA e conhecida (N = numero de municipios),
%  e n0 pode dar maior que N (o que e logicamente impossivel), aplicamos
%  tambem a correcao de populacao finita (FPC):
%       n = n0 / (1 + (n0-1)/N)
%
%  Compativel com MATLAB e GNU Octave (sem toolbox: z tabelado na mao).
% ========================================================================
clear all; close all; clc;

%% --- 0) Parametros do problema -----------------------------------------
ANO_ALVO   = 2024;     % <-- ajuste aqui se o professor pediu outro ano
confianca  = 0.98;     % 98% de confianca
E          = 10000;    % erro maximo admitido (tCO2/ano)

% z_(alpha/2) para os niveis de confianca mais comuns (tabela normal padrao)
switch round(confianca*100)
    case 90, z = 1.6449;
    case 95, z = 1.9600;
    case 98, z = 2.3263;
    case 99, z = 2.5758;
    otherwise
        error('Nivel de confianca %.2f%% nao esta na tabela embutida.', 100*confianca);
end

%% --- 1) Leitura dos dados e estatisticas populacionais (igual a 1.1) ---
arquivo_csv = 'dados_co2_municipios.csv';
fid = fopen(arquivo_csv, 'r');
if fid == -1
    error('Nao foi possivel abrir "%s".', arquivo_csv);
end
fgetl(fid);
C = textscan(fid, '%q %f %f %f %f %f %f %f %f', 'Delimiter', ',');
fclose(fid);
CO2  = cell2mat(C(2:9));
anos = [1990 2000 2010 2020 2021 2022 2023 2024];

N = size(CO2, 1);                 % tamanho da populacao (numero de municipios)
media_pop  = mean(CO2, 1);
desvio_pop = std(CO2, 1, 1);      % desvio padrao POPULACIONAL (N)

%% --- 2) Tamanho de amostra minimo para o ano alvo ----------------------
idx = find(anos == ANO_ALVO, 1);
if isempty(idx)
    error('Ano %d nao esta na lista de anos disponiveis.', ANO_ALVO);
end
sigma = desvio_pop(idx);

n0    = (z * sigma / E)^2;                  % formula classica (pop. infinita)
n_fpc = n0 / (1 + (n0 - 1)/N);              % com correcao de populacao finita

fprintf('=== Questao 1.4 - Ano de referencia: %d ===\n', ANO_ALVO);
fprintf('Desvio padrao populacional (sigma) = %.2f tCO2/ano\n', sigma);
fprintf('Numero total de municipios (N)     = %d\n', N);
fprintf('z (%.0f%% de confianca)              = %.4f\n', 100*confianca, z);
fprintf('Erro maximo admitido (E)           = %.2f tCO2/ano\n\n', E);

fprintf('n0    (sem correcao, pop. infinita) = %.2f  -> arredondando: n0    = %d\n', n0, ceil(n0));
fprintf('n_fpc (com correcao de pop. finita) = %.2f  -> arredondando: n_fpc = %d\n', n_fpc, ceil(n_fpc));

if n0 > N
    fprintf(['\nObs.: n0 > N (%.0f > %d), ou seja, a formula classica pediria mais\n' ...
             'amostras do que municipios existem. Isso e um sinal claro de que a\n' ...
             'correcao de populacao finita e necessaria aqui; a resposta correta\n' ...
             'e n_fpc = %d municipios (%.1f%% da populacao total).\n'], ...
             n0, N, ceil(n_fpc), 100*n_fpc/N);
end

%% --- 3) Para referencia: mesmo calculo para todos os anos --------------
fprintf('\n=== Para referencia: mesmo calculo em todos os anos ===\n');
fprintf('%6s %14s %14s %14s\n', 'Ano', 'Sigma', 'n0', 'n_fpc');
for k = 1:numel(anos)
    s_k = desvio_pop(k);
    n0_k  = (z * s_k / E)^2;
    nfpc_k = n0_k / (1 + (n0_k - 1)/N);
    fprintf('%6d %14.2f %14.0f %14.0f\n', anos(k), s_k, ceil(n0_k), ceil(nfpc_k));
end