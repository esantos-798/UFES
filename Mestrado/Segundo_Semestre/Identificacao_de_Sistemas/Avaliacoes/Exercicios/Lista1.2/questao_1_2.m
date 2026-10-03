%% ========================================================================
%  Questao 1.2 (complemento) - Teste de estacionariedade grafico
%  (media vs. amplitude / media vs. desvio padrao)
%
%  Adaptacao do teste classico (originalmente feito com blocos consecutivos
%  de uma unica realizacao) para o nosso caso: aqui cada "subconjunto" S^k
%  e o conjunto de todos os municipios em um dado ANO k (k = 1..8), em vez
%  de um bloco de observacoes consecutivas de uma serie temporal.
%
%  Se w_k (amplitude) ou o desvio padrao forem independentes de X_barra_k
%  (media), os pontos devem se espalhar ao redor de uma reta horizontal
%  -> compativel com estacionariedade (na dispersao). Caso contrario, ha
%  indicio de heterocedasticidade (variancia dependente do nivel).
%
%  Compativel com MATLAB e GNU Octave. Sem funcoes locais (algumas
%  instalacoes de Octave nao reconhecem funcao local no fim de um script).
% ========================================================================
clear all; close all; clc;

%% --- 1) Leitura dos dados --------------------------------------------
arquivo_csv = 'dados_co2_municipios.csv';
[~, CO2] = prepara_dados_entrada(arquivo_csv);
anos = [1990 2000 2010 2020 2021 2022 2023 2024];

%% --- 2) Media, amplitude (range) e desvio padrao por ano --------------
media_ano = mean(CO2, 1);
range_ano = max(CO2, [], 1) - min(CO2, [], 1);   % w_k = amplitude
std_ano   = std(CO2, 1, 1);                       % desvio padrao populacional

fprintf('%6s %16s %16s %16s\n', 'Ano', 'Media', 'Amplitude(w_k)', 'DesvioPadrao');
for k = 1:numel(anos)
    fprintf('%6d %16.0f %16.0f %16.0f\n', anos(k), media_ano(k), range_ano(k), std_ano(k));
end

% Coeficiente de correlacao de Pearson calculado na mao (sem corr()/corrcoef(),
% que exigem o Statistics Toolbox / pacote statistics do Octave)
xm = media_ano - mean(media_ano);

ym1 = range_ano - mean(range_ano);
r_range = sum(xm .* ym1) / sqrt(sum(xm.^2) * sum(ym1.^2));

ym2 = std_ano - mean(std_ano);
r_std = sum(xm .* ym2) / sqrt(sum(xm.^2) * sum(ym2.^2));

fprintf('\nCorrelacao (media, amplitude)     = %.3f\n', r_range);
fprintf('Correlacao (media, desvio padrao) = %.3f\n', r_std);

%% --- 3) Graficos: media (eixo x) vs. dispersao (eixo y) ----------------
figure('Name', 'Teste de estacionariedade - media vs. amplitude/desvio');

subplot(1,2,1);
plot(media_ano, range_ano, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
for k = 1:numel(anos)
    text(media_ano(k), range_ano(k), sprintf('  %d', anos(k)));
end
xlabel('Media X_k (tCO2/ano)', 'Interpreter', 'none');
ylabel('Amplitude w_k = max-min (tCO2/ano)', 'Interpreter', 'none');
title(sprintf('Media vs. Amplitude (r = %.2f)', r_range), 'Interpreter', 'none');
grid on;

subplot(1,2,2);
plot(media_ano, std_ano, 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
for k = 1:numel(anos)
    text(media_ano(k), std_ano(k), sprintf('  %d', anos(k)));
end
xlabel('Media X_k (tCO2/ano)', 'Interpreter', 'none');
ylabel('Desvio padrao (tCO2/ano)', 'Interpreter', 'none');
title(sprintf('Media vs. Desvio padrao (r = %.2f)', r_std), 'Interpreter', 'none');
grid on;

drawnow;
quadro = getframe(gcf);
imwrite(quadro.cdata, 'teste_estacionariedade_1_2.png');
fprintf('\nFigura salva: teste_estacionariedade_1_2.png\n');