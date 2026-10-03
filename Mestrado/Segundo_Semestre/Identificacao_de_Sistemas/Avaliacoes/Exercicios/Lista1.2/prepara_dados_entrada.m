function [cidades, CO2] = prepara_dados_entrada(arquivo_csv)
%% ---  Leitura dos dados ------------------------------------------------
% O arquivo original "Dados-municipais.xlsx" tem ~250 MB porque a aba
% "Dados" traz o detalhe por setor/subsetor de emissao (mais de 350 mil
% linhas). Para a questao 1.1 so precisamos do TOTAL de CO2 por municipio
% em cada ano, que ja vem pronto na aba "Consulta Ranking" (tabela
% dinamica) do proprio arquivo.
%
% Em vez de depender do pacote "io" do Octave para ler .xlsx (que por sua
% vez depende de Perl/Java, nem sempre disponiveis), essa aba foi
% exportada uma unica vez para o arquivo texto "dados_co2_municipios.csv"
% (Municipio, e uma coluna por ano). O script abaixo le esse .csv com
% textscan, que existe nativamente tanto no MATLAB quanto no Octave.

arquivo_csv = 'dados_co2_municipios.csv';

fid = fopen(arquivo_csv, 'r');
if fid == -1
    error('Nao foi possivel abrir "%s". Verifique se ele esta na mesma pasta do script.', arquivo_csv);
end
fgetl(fid);   % descarta a linha de cabecalho
C = textscan(fid, '%q %f %f %f %f %f %f %f %f', 'Delimiter', ',');
fclose(fid);

cidades = C{1};                 % cell array de strings (nomes dos municipios)
CO2     = cell2mat(C(2:9));     % matriz N_municipios x 8 anos [tCO2/ano]
