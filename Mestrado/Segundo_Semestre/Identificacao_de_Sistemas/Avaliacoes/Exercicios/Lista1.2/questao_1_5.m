%% ========================================================================
%  Questao 1.5 - Teste de hipotese: a media de emissao das 8 cidades do
%  aluno em 2024 difere da media geral (populacional) de 2024?
%
%  H0: mu_amostra = mu_populacao   (a media das 8 cidades = media geral)
%  H1: mu_amostra ~= mu_populacao  (teste bicaudal)
%
%  Como o desvio padrao POPULACIONAL (sigma) e conhecido (calculado na
%  1.1), o teste apropriado e o teste Z para uma amostra:
%
%       z = (x_barra - mu) / (sigma / sqrt(n))
%
%  Compativel com MATLAB e GNU Octave. p-valor calculado via erf(), que e
%  funcao nativa (nao precisa de normcdf()/Statistics Toolbox):
%       normcdf(z) = 0.5*(1 + erf(z/sqrt(2)))
% ========================================================================
clear all; close all; clc;

%% --- 0) Parametros -------------------------------------------------------
ANO_ALVO = 2024;
alpha    = 0.05;   % nivel de significancia (nao especificado no enunciado;
                    % 5% e a convencao padrao - ajuste aqui se necessario)

%% --- 1) Estatisticas POPULACIONAIS do ano alvo (igual a 1.1) --------------
arquivo_csv = 'dados_co2_municipios.csv';
fid = fopen(arquivo_csv, 'r');
if fid == -1
    error('Nao foi possivel abrir "%s".', arquivo_csv);
end
fgetl(fid);
C = textscan(fid, '%q %f %f %f %f %f %f %f %f', 'Delimiter', ',');
fclose(fid);
todas_cidades = C{1};
todos_CO2     = cell2mat(C(2:9));
anos          = [1990 2000 2010 2020 2021 2022 2023 2024];

idx_ano = find(anos == ANO_ALVO, 1);
mu    = mean(todos_CO2(:, idx_ano));
sigma = std(todos_CO2(:, idx_ano), 1);   % desvio padrao POPULACIONAL (N)

%% --- 2) Media amostral das 8 cidades do aluno no ano alvo -----------------
minhas_cidades = {
    'Salvador (BA)'; 'São Paulo (SP)'; 'Rio de Janeiro (RJ)'; 'Porto Alegre (RS)';
    'Teresina (PI)'; 'Maceió (AL)'; 'Goiânia (GO)'; 'Boa Vista (RR)'};
n = numel(minhas_cidades);

valores_amostra = zeros(n, 1);
for i = 1:n
    idx_cidade = find(strcmp(todas_cidades, minhas_cidades{i}));
    valores_amostra(i) = todos_CO2(idx_cidade, idx_ano);
end
x_barra = mean(valores_amostra);

fprintf('=== Questao 1.5 - Teste Z para uma amostra (ano %d) ===\n\n', ANO_ALVO);
fprintf('Media populacional (mu)      = %.2f tCO2/ano  (N = %d municipios)\n', mu, size(todos_CO2,1));
fprintf('Desvio padrao populacional (sigma) = %.2f tCO2/ano\n', sigma);
fprintf('Media amostral (x_barra, n=%d)     = %.2f tCO2/ano\n\n', n, x_barra);

%% --- 3) Estatistica de teste e p-valor -------------------------------------
erro_padrao = sigma / sqrt(n);
z = (x_barra - mu) / erro_padrao;

% p-valor bicaudal via funcao erro (erf), sem depender de normcdf()
normcdf_z = 0.5 * (1 + erf(z / sqrt(2)));
p_valor = 2 * (1 - normcdf_z);
if p_valor < 0
    p_valor = 0;  % protege contra erro numerico para |z| muito grande
end

% valor critico bicaudal para o alpha escolhido (tabela normal padrao)
switch round((1-alpha)*100)
    case 90, z_critico = 1.6449;
    case 95, z_critico = 1.9600;
    case 98, z_critico = 2.3263;
    case 99, z_critico = 2.5758;
    otherwise
        z_critico = NaN;
        warning('alpha = %.3f nao esta na tabela embutida; comparando so pelo p-valor.', alpha);
end

fprintf('Erro padrao (sigma/sqrt(n))  = %.2f tCO2/ano\n', erro_padrao);
fprintf('Estatistica de teste z       = %.4f\n', z);
fprintf('p-valor (teste bicaudal)     = %.6g\n', p_valor);
if ~isnan(z_critico)
    fprintf('Valor critico |z| (alpha=%.2f) = %.4f\n', alpha, z_critico);
end

%% --- 4) Decisao ------------------------------------------------------------
fprintf('\n--- Decisao (alpha = %.2f) ---\n', alpha);
if p_valor < alpha
    fprintf(['Rejeita-se H0: ha evidencia estatisticamente significativa de que a\n' ...
             'media de emissao das 8 cidades do aluno difere da media geral dos\n' ...
             'municipios brasileiros em %d.\n'], ANO_ALVO);
else
    fprintf(['Nao se rejeita H0: nao ha evidencia estatisticamente significativa de\n' ...
             'diferenca entre a media das 8 cidades do aluno e a media geral em %d.\n'], ANO_ALVO);
end
