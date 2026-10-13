% ==========================================================
%  MODELAGEM DE RESPOSTA AO IMPULSO
%  - Underwater Wireless Optical Communication (UWOC) -
%  VERSÃO VETORIZADA DO NÚCLEO DE MONTE CARLO
% ==========================================================
%  Universidade Federal do Espírito Santo
%  Programa de Pós-Graduação em Engenharia Elétrica
%
%  Cristiano Tavares                 Jair Adriano Silva Lima
%                                                 Julho/2026
% ==========================================================
%
%  Esta versão substitui o laço "for fóton a fóton" original por
%  um laço vetorizado sobre TODOS os fótons ainda ativos a cada
%  iteração. A física (Lambertiano generalizado, extinção,
%  peso/albedo estilo Tang, espalhamento Henyey-Greenstein) é
%  idêntica à versão original — só a forma de calcular mudou.
%
%  Ganho esperado: a versão original faz ~800 milhões de
%  iterações escalares de while (20e6 fótons x ~40 passos cada).
%  Aqui o número de iterações do laço cai para algumas dezenas
%  (~40-70), cada uma operando sobre vetores/matrizes de até
%  20 milhões de linhas de uma vez.
%
%  Atenção à memória: pos/dir usam N x 3 doubles (~480 MB cada
%  para N = 20e6); W, dist_total e os buffers de saída usam
%  N x 1 doubles (~160 MB cada). Total ~1.5-2 GB de RAM para
%  N = 20e6. Se isso for um problema na sua máquina, processe em
%  blocos (ver comentário "PROCESSAMENTO EM BLOCOS" no final).
% ==========================================================

clear; clc; close all;
rng(1);     % fixa a sequência de números aleatórios

% ==========================================================
% Parâmetros de entrada
% ==========================================================
num_fotons = 20e6;          % Número de fótons
L = 50;                     % Distância
a = 0.088;                  % Absorção (Coastal)
b = 0.216;                  % Espalhamento (Coastal)
v = 2.25e8;                 % Velocidade da luz na água
g = 0.9470;                 % Fator HG (Henyey–Greenstein)
raio_detector = 0.25;       % Receptor [m] (Tang)
angulo_divergencia = 10;    % Divergência total da fonte [graus] (Tang)
W_min = 1e-6;               % Peso mínimo considerado na degradação

janela_ns = 16;              % Janela de observação
passo_s = 0.1e-9;            % Passo do histograma de Monte Carlo

% ----------------------------------------------------------
% Cálculos iniciais
% ----------------------------------------------------------
c = a + b;                  % Coeficiente de extinção
albedo = 1 - a/c;           % = b/c, fator de sobrevivência por colisão

theta_half = deg2rad(angulo_divergencia/2);
m_led = -log(2) / log(cos(theta_half));   % Ordem Lambertiana

maxiter_seguranca = 2000;   % trava de segurança contra loop infinito

% ==========================================================
% Simulação de Monte Carlo (VETORIZADA)
% ==========================================================

% --- Buffers de saída pré-alocados (tamanho máximo possível) ---
t_chegada_buf    = zeros(num_fotons, 1);
peso_chegada_buf = zeros(num_fotons, 1);
n_hits = 0;

% --- Estado inicial de todos os fótons de uma vez ---
pos        = zeros(num_fotons, 3);
dist_total = zeros(num_fotons, 1);
W          = ones(num_fotons, 1);

R_theta = rand(num_fotons, 1);
theta0  = acos((1 - R_theta) .^ (1/(m_led + 1)));
phi0    = 2*pi*rand(num_fotons, 1);

dir = [cos(theta0), sin(theta0).*cos(phi0), sin(theta0).*sin(phi0)];
dir = dir ./ vecnorm(dir, 2, 2);

ativo = true(num_fotons, 1);   % fótons ainda em propagação
n_ativos_inicial = num_fotons;

WB = waitbar(0, 'Executando...', 'Name', 'Simulação Monte Carlo (vetorizada)');

iter = 0;
while any(ativo)
    iter = iter + 1;
    if iter > maxiter_seguranca
        warning('Número máximo de iterações atingido (%d); interrompendo.', maxiter_seguranca);
        break;
    end

    idx = find(ativo);
    n   = numel(idx);

    pos_a = pos(idx, :);
    dir_a = dir(idx, :);
    W_a   = W(idx);
    dt_a  = dist_total(idx);

    % 1) sorteia a distância até a próxima interação
    dist_passo = -log(rand(n, 1)) / c;

    % 2) verifica cruzamento com o plano receptor x = L
    dir_pos = dir_a(:, 1) > 0;
    dist_ate_rx = inf(n, 1);
    dist_ate_rx(dir_pos) = (L - pos_a(dir_pos, 1)) ./ dir_a(dir_pos, 1);

    cruzou = dir_pos & dist_ate_rx >= 0 & dist_ate_rx <= dist_passo;

    if any(cruzou)
        pos_hit = pos_a(cruzou, :) + dir_a(cruzou, :) .* dist_ate_rx(cruzou);
        y_hit = pos_hit(:, 2);
        z_hit = pos_hit(:, 3);
        no_detector = (y_hit.^2 + z_hit.^2) <= raio_detector^2;

        dist_chegada = dt_a(cruzou) + dist_ate_rx(cruzou);
        t_hit = dist_chegada / v;
        w_hit = W_a(cruzou);

        t_hit = t_hit(no_detector);
        w_hit = w_hit(no_detector);

        k = numel(t_hit);
        if k > 0
            t_chegada_buf(n_hits+1 : n_hits+k)    = t_hit;
            peso_chegada_buf(n_hits+1 : n_hits+k) = w_hit;
            n_hits = n_hits + k;
        end
    end

    % 3) fótons que NÃO cruzaram o plano: percorrem o passo completo
    continua = ~cruzou;

    pos_a(continua, :) = pos_a(continua, :) + dir_a(continua, :) .* dist_passo(continua);
    dt_a(continua)     = dt_a(continua) + dist_passo(continua);

    % Absorção: degradação de peso (fórmula Tang / albedo de espalhamento)
    W_a(continua) = W_a(continua) * albedo;

    % Critérios de término: peso muito baixo ou distância excessiva
    morreu_peso = continua & (W_a < W_min);
    morreu_dist = continua & (dt_a >= 5*L);
    sobrevive   = continua & ~morreu_peso & ~morreu_dist;

    % 4) espalhamento Henyey-Greenstein (só para quem sobreviveu)
    if any(sobrevive)
        ns = nnz(sobrevive);
        r  = rand(ns, 1);
        cos_theta = (1/(2*g)) * (1 + g^2 - ((1 - g^2) ./ (1 - g + 2*g*r)).^2);
        cos_theta = max(-1, min(1, cos_theta));
        sin_theta = sqrt(1 - cos_theta.^2);
        phi = 2*pi*rand(ns, 1);

        u = dir_a(sobrevive, :);

        % vetor perpendicular a u (evita degenerescência perto do eixo z)
        e1 = zeros(ns, 3);
        perto_z = abs(u(:, 3)) >= 0.999;
        e1(~perto_z, :) = cross(repmat([0 0 1], nnz(~perto_z), 1), u(~perto_z, :), 2);
        e1(perto_z, :)  = cross(repmat([0 1 0], nnz(perto_z), 1),  u(perto_z, :),  2);
        e1 = e1 ./ vecnorm(e1, 2, 2);

        e2 = cross(u, e1, 2);

        nova_dir = cos_theta .* u + sin_theta .* cos(phi) .* e1 + sin_theta .* sin(phi) .* e2;
        dir_a(sobrevive, :) = nova_dir ./ vecnorm(nova_dir, 2, 2);
    end

    % --- grava de volta o estado atualizado ---
    pos(idx, :)     = pos_a;
    dir(idx, :)     = dir_a;
    W(idx)          = W_a;
    dist_total(idx) = dt_a;

    % desativa quem cruzou o plano ou morreu por peso/distância
    idx_termina = idx(cruzou | morreu_peso | morreu_dist);
    ativo(idx_termina) = false;

    % --- atualização da barra de progresso ---
    if mod(iter, 5) == 0 || ~any(ativo)
        frac_concluida = 1 - nnz(ativo)/n_ativos_inicial;
        waitbar(frac_concluida, WB, sprintf('%.1f%% concluído (iteração %d)', 100*frac_concluida, iter));
    end
end
close(WB)

t_chegada    = t_chegada_buf(1:n_hits).';
peso_chegada = peso_chegada_buf(1:n_hits).';

clear pos dir W dist_total ativo t_chegada_buf peso_chegada_buf pos_a dir_a W_a dt_a

% ==========================================================
% DAQUI PARA BAIXO: EXATAMENTE IGUAL AO SCRIPT ORIGINAL
% (filtragem por janela, histograma, ajuste Double Gamma,
%  R²/RMSE, desnormalização e gráfico)
% ==========================================================

% Tratamento dos dados de saída
t0 = L / v;
t_limite = t0 + (janela_ns * 1e-9);

validos = (t_chegada <= t_limite);
t_chegada_valido    = t_chegada(validos);
peso_chegada_valido = peso_chegada(validos);

% Criação das caixas de tempo do histograma
t_bins = t0 : passo_s : t_limite;
t_centro = t_bins(1:end-1) + passo_s/2;

% ======================================================
% Montagem do histograma
% ======================================================
[~, ~, bin] = histcounts(t_chegada_valido, t_bins);
idx_bin = bin > 0;

contagens = accumarray(bin(idx_bin).', peso_chegada_valido(idx_bin).', [length(t_bins)-1, 1], @sum, 0).';

h_mc = contagens / num_fotons;

h_max = max(h_mc);
if h_max == 0
    error('NENHUM fóton atingiu o receptor!');
end

t0_ref = min(t_bins);
t_data_ns = (t_centro - t0_ref) * 1e9;
h_mc_norm = h_mc / h_max;

% ==========================================================
% Modelo Double Gamma Functions - DGF
% ==========================================================
dgf_norm = @(C, t) (t >= 0) .* (C(1).*t.*exp(-C(2).*t) + C(3).*t.*exp(-C(4).*t));

C0 = [1, 0.5, 1, 0.5];
lb = [0, 0, 0, 0];
ub = [10, 10, 10, 10];

options = optimoptions('lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'Display', 'iter', 'FunctionTolerance', 1e-12);

C_finais = lsqcurvefit(dgf_norm, C0, t_data_ns, h_mc_norm, lb, ub, options);

% ==========================================================
% Avaliação do ajuste - R² e RMSE
% ==========================================================
h_dgf_norm = dgf_norm(C_finais, t_data_ns);

erro = h_mc_norm - h_dgf_norm;
RMSE = sqrt(mean(erro.^2));

SS_res = sum((h_mc_norm - h_dgf_norm).^2);
SS_tot = sum((h_mc_norm - mean(h_mc_norm)).^2);
R2 = 1 - (SS_res / SS_tot);

% ==========================================================
% Exibição dos resultados
% ==========================================================
fprintf('\n=== INFORMAÇÕES DA SIMULAÇÃO ===\n');
fprintf('Fótons emitidos: %d\n', num_fotons);
fprintf('Fótons detectados: %d\n', length(t_chegada));
fprintf('Soma dos pesos recebidos: %.6e\n', sum(peso_chegada));
fprintf('Ordem Lambertiana m = %.2f\n', m_led);

fprintf('\n=== QUALIDADE DO AJUSTE DOUBLE GAMMA ===\n');
fprintf('R²   = %.6f\n', R2);
fprintf('RMSE = %.6f\n', RMSE);

fprintf('\n=== COEFICIENTES NORMALIZADOS DA GAMA DUPLA ===\n');
fprintf('C1 = %.6f\n', C_finais(1));
fprintf('C2 = %.6f\n', C_finais(2));
fprintf('C3 = %.6f\n', C_finais(3));
fprintf('C4 = %.6f\n', C_finais(4));

escala_tempo = 1e9;
C_original_1 = C_finais(1) * h_max * escala_tempo;
C_original_2 = C_finais(2) * escala_tempo;
C_original_3 = C_finais(3) * h_max * escala_tempo;
C_original_4 = C_finais(4) * escala_tempo;

fprintf('\n=== COEFICIENTES REAIS DA GAMA DUPLA ===\n');
fprintf('C_original_1 = %.6f\n', C_original_1);
fprintf('C_original_2 = %.6f\n', C_original_2);
fprintf('C_original_3 = %.6f\n', C_original_3);
fprintf('C_original_4 = %.6f\n', C_original_4);

% ==========================================================
% Gráfico final
% ==========================================================
delta_t_segundos = t_centro - t0_ref;
h_verificacao = (C_original_1 .* delta_t_segundos .* exp(-C_original_2 .* delta_t_segundos)) + ...
    (C_original_3 .* delta_t_segundos .* exp(-C_original_4 .* delta_t_segundos));

figure('Color', 'w', 'Name', 'Resposta ao Impulso');
bar(delta_t_segundos * 1e9, h_mc, 'DisplayName', 'Monte Carlo');
hold on;
plot(delta_t_segundos * 1e9, h_verificacao, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Modelo Gama Dupla');

xlabel('Tempo após t_0 (ns)', 'FontWeight', 'bold');
ylabel('Resposta ao Impulso do Canal [h(t)]', 'FontWeight', 'bold');
title('Ajuste do Modelo de Canal UWOC');
legend;
grid on;

% ==========================================================
% PROCESSAMENTO EM BLOCOS (se faltar memória para 20e6 de uma vez)
% ==========================================================
% Se num_fotons for grande demais para a RAM disponível, basta
% envolver todo o bloco "Simulação de Monte Carlo (VETORIZADA)"
% em um laço externo, processando por exemplo 2e6 fótons por vez
% e acumulando t_chegada/peso_chegada (ou diretamente as
% 'contagens' do histograma) entre os blocos, em vez de alocar
% os 20e6 de uma só vez. A física do laço interno não muda.