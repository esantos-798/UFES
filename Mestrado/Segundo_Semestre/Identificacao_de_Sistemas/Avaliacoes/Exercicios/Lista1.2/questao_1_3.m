%% ========================================================================
%  Questao 1.3 - Estatisticas amostrais (n = 8 cidades do aluno) por ano:
%  media + intervalo de confianca, mediana, desvio padrao, coeficiente de
%  variacao, distribuicao de frequencia, histograma e box-plot.
%
%  Compativel com MATLAB e GNU Octave. Nao depende de nenhum toolbox/
%  pacote extra (nem Statistics Toolbox nem "statistics" do Octave):
%  tinv() e boxplot() sao reimplementados manualmente abaixo, ja que a
%  instalacao usada neste exercicio teve varios problemas com pacotes
%  externos (io, gnuplot, Ghostscript etc.) nas questoes anteriores.
%
%  Autor: Du (I = 5)
% ========================================================================
clear all; close all; clc;

%% --- 1) Leitura dos dados e selecao das 8 cidades do aluno ----------------
arquivo_csv = 'dados_co2_municipios.csv';
fid = fopen(arquivo_csv, 'r');
if fid == -1
    error('Nao foi possivel abrir "%s". Verifique se ele esta na mesma pasta do script.', arquivo_csv);
end
fgetl(fid);
C = textscan(fid, '%q %f %f %f %f %f %f %f %f', 'Delimiter', ',');
fclose(fid);

todas_cidades = C{1};
todos_CO2     = cell2mat(C(2:9));
anos          = [1990 2000 2010 2020 2021 2022 2023 2024];

% 8 cidades associadas ao ID = 5 (ver Tabela 1 do enunciado / Minhas_cidades.txt)
minhas_cidades = {
    'Salvador (BA)'         % #1
    'São Paulo (SP)'        % #2
    'Rio de Janeiro (RJ)'   % #3
    'Porto Alegre (RS)'     % #4
    'Teresina (PI)'         % #5
    'Maceió (AL)'           % #6
    'Goiânia (GO)'          % #7
    'Boa Vista (RR)'        % #8 - atencao: existe tambem "Boa Vista (PB)"
};

n = numel(minhas_cidades);
CO2_amostra = nan(n, numel(anos));
for i = 1:n
    idx = find(strcmp(todas_cidades, minhas_cidades{i}));
    if isempty(idx)
        error('Cidade "%s" nao encontrada em %s.', minhas_cidades{i}, arquivo_csv);
    end
    CO2_amostra(i, :) = todos_CO2(idx, :);
end

fprintf('Cidades da amostra (n = %d):\n', n);
for i = 1:n
    fprintf('  #%d - %s\n', i, minhas_cidades{i});
end
fprintf('\n');

%% --- 2) Media, mediana, desvio padrao, CV e intervalo de confianca --------
nivel_confianca = 0.95;   % <-- ajuste aqui se o professor pedir outro nivel
alpha = 1 - nivel_confianca;

media   = mean(CO2_amostra, 1);          % media amostral
mediana = median(CO2_amostra, 1);        % mediana
desvio  = std(CO2_amostra, 0, 1);        % desvio padrao AMOSTRAL (N-1)
cv      = 100 * desvio ./ media;         % coeficiente de variacao (%)
erro_padrao = desvio / sqrt(n);

% Valor critico Z (NAO t-Student): o material de Estatistica do curso
% (transparencia da Fucape, Prof. Fernando Caio Galdi) usa Z em todos os
% exemplos de intervalo de confianca, mesmo com amostras pequenas - nao
% ha nenhuma mencao a distribuicao t/Student nesse material. Seguimos a
% mesma convencao aqui, usando o desvio padrao amostral s no lugar de
% sigma (igual aos exemplos do material, que sempre tratam o desvio
% informado como se fosse o parametro populacional).
switch round(nivel_confianca * 100)
    case 90, z_critico = 1.645;
    case 95, z_critico = 1.960;
    case 98, z_critico = 2.326;
    case 99, z_critico = 2.575;
    otherwise
        error('nivel_confianca = %.2f nao esta na tabela embutida (use 0.90, 0.95, 0.98 ou 0.99).', nivel_confianca);
end

margem = z_critico * erro_padrao;
IC_inf = media - margem;
IC_sup = media + margem;

fprintf('=== Estatisticas amostrais por ano (n = %d, IC %.0f%%) ===\n', n, 100*nivel_confianca);
fprintf('%6s %14s %14s %14s %9s %16s %16s\n', ...
    'Ano', 'Media', 'Mediana', 'DesvPad', 'CV(%)', 'IC_inf', 'IC_sup');
for k = 1:numel(anos)
    fprintf('%6d %14.2f %14.2f %14.2f %9.2f %16.2f %16.2f\n', ...
        anos(k), media(k), mediana(k), desvio(k), cv(k), IC_inf(k), IC_sup(k));
end
fprintf('\n');

%% --- 3) Distribuicao de frequencias (regra de Sturges: k = 1+3.322*log10(n)) --
k_classes = max(3, round(1 + 3.322 * log10(n)));  % n=8 -> k=4
fprintf('=== Distribuicao de frequencias (%d classes, regra de Sturges) ===\n', k_classes);
for j = 1:numel(anos)
    x = CO2_amostra(:, j);
    minx = min(x); maxx = max(x);
    bordas = linspace(minx, maxx, k_classes + 1);
    contagem = histc(x, bordas);
    contagem(end-1) = contagem(end-1) + contagem(end);  % junta ultimo limite superior
    contagem(end) = [];

    fprintf('\nAno %d:\n', anos(j));
    fprintf('  %22s  %10s  %12s\n', 'Classe (tCO2/ano)', 'Freq.', 'Freq. rel.(%)');
    for c = 1:k_classes
        fprintf('  [%10.1f ; %10.1f)  %10d  %12.1f\n', ...
            bordas(c), bordas(c+1), contagem(c), 100*contagem(c)/n);
    end
end
fprintf('\n');

%% --- 4) Histogramas (um subplot por ano) ----------------------------------
figure('Name', 'Questao 1.3 - Histogramas da emissao de CO2 (8 cidades do aluno)');
for k = 1:numel(anos)
    x = CO2_amostra(:, k) / 1e6;   % exibe em milhoes de tCO2 (rotulos mais limpos)
    subplot(2, 4, k);
    if exist('histogram', 'file') == 2 || exist('histogram', 'builtin') ~= 0
        histogram(x, k_classes);
    else
        hist(x, k_classes);
    end
    title(sprintf('Ano %d', anos(k)), 'Interpreter', 'none');
    xlabel('CO2 (milhoes de tCO2/ano)', 'Interpreter', 'none');
    ylabel('No. de cidades', 'Interpreter', 'none');
    xtickangle(45);
    grid on;
end
drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'hist_1_3_amostra.png');
fprintf('Figura salva: hist_1_3_amostra.png\n');

%% --- 5) Box-plot (uma caixa por ano) ---------------------------------------
% Implementado manualmente (sem depender de boxplot()/Statistics Toolbox):
% quartis calculados por interpolacao linear (interp1), que e funcao
% nativa tanto do MATLAB quanto do Octave.
figure('Name', 'Questao 1.3 - Box-plot da emissao de CO2 por ano (8 cidades do aluno)');
hold on;
largura = 0.3;
for k = 1:numel(anos)
    x = sort(CO2_amostra(:, k));
    pos_quartis = 1 + [0.25 0.50 0.75] * (n - 1);
    quartis = interp1(1:n, x, pos_quartis);
    Q1 = quartis(1); Q2 = quartis(2); Q3 = quartis(3);
    IQR = Q3 - Q1;

    limite_inf = max(min(x), Q1 - 1.5*IQR);
    limite_sup = min(max(x), Q3 + 1.5*IQR);
    outliers = x(x < limite_inf | x > limite_sup);

    % caixa (retangulo Q1-Q3)
    rectangle('Position', [k - largura/2, Q1, largura, Q3 - Q1], ...
              'EdgeColor', 'k', 'FaceColor', [0.8 0.9 1]);
    % mediana
    line([k - largura/2, k + largura/2], [Q2 Q2], 'Color', 'r', 'LineWidth', 2);
    % whiskers
    line([k k], [Q3 limite_sup], 'Color', 'k');
    line([k k], [Q1 limite_inf], 'Color', 'k');
    line([k - largura/4, k + largura/4], [limite_sup limite_sup], 'Color', 'k');
    line([k - largura/4, k + largura/4], [limite_inf limite_inf], 'Color', 'k');
    % outliers
    if ~isempty(outliers)
        plot(k * ones(size(outliers)), outliers, 'ro', 'MarkerSize', 5);
    end
end
hold off;
set(gca, 'XTick', 1:numel(anos), 'XTickLabel', arrayfun(@num2str, anos, 'UniformOutput', false));
xlabel('Ano');
ylabel('Emissao de CO2 (tCO2/ano)');
title('Box-plot por ano - 8 cidades do aluno (I = 5)');
grid on;
drawnow;
quadro2 = getframe(gcf);
imwrite(quadro2.cdata, 'boxplot_1_3_amostra.png');
fprintf('Figura salva: boxplot_1_3_amostra.png\n');

%% --- 6) Salvar resultados para uso nos proximos itens ----------------------
save('resultado_questao1_3.mat', 'anos', 'minhas_cidades', 'CO2_amostra', ...
     'media', 'mediana', 'desvio', 'cv', 'IC_inf', 'IC_sup', 'n');