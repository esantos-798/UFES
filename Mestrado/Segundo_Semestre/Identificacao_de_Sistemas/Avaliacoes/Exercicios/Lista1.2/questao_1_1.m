%% ========================================================================
%  Questao 1.1 - Media, desvio padrao e histograma da emissao de CO2
%  (populacao = todos os municipios brasileiros), para os anos
%  1990, 2000, 2010, 2020, 2021, 2022, 2023 e 2024.
%
%  Compativel com MATLAB e GNU Octave (sem pacotes extras: nao usa
%  readtable/xlsread, so funcoes nativas de I/O de texto).
%
%  Autor: Eduardo Santos
% ========================================================================
clear all; close all; clc;
%% --- 1) Leitura dos dados ----------------
arquivo_csv = 'dados_co2_municipios.csv';
[cidades, CO2] = prepara_dados_entrada(arquivo_csv);
anos    = [1990 2000 2010 2020 2021 2022 2023 2024];

N = size(CO2, 1);
fprintf('Numero de municipios (N = tamanho da populacao) = %d\n\n', N);

%% --- 2) Media e desvio padrao POPULACIONAIS (item 1.1) --------------------
% Como o enunciado pede a MEDIA POPULACIONAL, o desvio padrao tambem deve
% ser o desvio padrao populacional (normalizado por N, e nao por N-1).
% Em MATLAB/Octave: std(x,1) usa normalizacao por N; std(x,0) (ou std(x))
% usa N-1 (amostral).

media_pop  = mean(CO2, 1);        % 1x8
desvio_pop = std(CO2, 1, 1);      % 1x8  (flag 1 = populacional, N)

fprintf('%6s %18s %22s\n', 'Ano', 'Media (tCO2)', 'Desvio padrao (tCO2)');
for k = 1:8
    fprintf('%6d %18.2f %22.2f\n', anos(k), media_pop(k), desvio_pop(k));
end

%% --- 3) Histogramas --------------------------------------------------
% A distribuicao de emissao por municipio e extremamente assimetrica: uns
% poucos municipios (grandes focos de desmatamento na Amazonia) emitem
% dezenas de milhoes de tCO2/ano, enquanto a maioria fica na casa dos
% milhares. Em escala linear isso faz quase tudo cair no primeiro bin e
% os rotulos do eixo x se sobrepoem. Por isso plotamos em log10(CO2),
% que e a forma padrao de visualizar dados com essa "cauda pesada".
% Municipios com emissao <= 0 (remocao liquida, poucos casos por ano) sao
% descartados apenas para esse grafico, pois log10 nao esta definido ali.

figure('Name', 'Questao 1.1 - Histogramas de log10(emissao de CO2) - todos os municipios');
nbins = 40;
for k = 1:8
    x = CO2(:, k);
    x = x(x > 0);              % remove nao positivos so para o log
    n_excluidos = N - numel(x);
    logx = log10(x);

    subplot(2, 4, k);
    if exist('histogram', 'file') == 2 || exist('histogram', 'builtin') ~= 0
        histogram(logx, nbins);
    else
        hist(logx, nbins);     % fallback para versoes antigas do Octave
    end
    title(sprintf('Ano %d (excl. %d)', anos(k), n_excluidos), 'Interpreter', 'none');
    xlabel('log10(CO2) [tCO2/ano]', 'Interpreter', 'none');
    ylabel('No. de municipios', 'Interpreter', 'none');
    grid on;
end
drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'hist_log10_co2.png');
fprintf('Figura salva: hist_log10_co2.png\n');

% Figura extra em escala linear (para referencia/comparacao), com menos
% ticks no eixo x para nao sobrepor os rotulos.
figure('Name', 'Questao 1.1 - Histogramas de emissao de CO2 (escala linear)');
for k = 1:8
    x = CO2(:, k);
    subplot(2, 4, k);
    if exist('histogram', 'file') == 2 || exist('histogram', 'builtin') ~= 0
        histogram(x, nbins);
    else
        hist(x, nbins);
    end
    title(sprintf('Ano %d', anos(k)), 'Interpreter', 'none');
    xlabel('Emissao de CO2 (tCO2/ano)', 'Interpreter', 'none');
    ylabel('No. de municipios', 'Interpreter', 'none');
    xtickangle(45);
    grid on;
end
drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'hist_linear_co2.png');
fprintf('Figura salva: hist_linear_co2.png\n');

%% --- 4) (opcional) salvar resultados para uso nos proximos itens ----------
save('resultado_questao1_1.mat', 'anos', 'cidades', 'CO2', 'media_pop', 'desvio_pop', 'N');