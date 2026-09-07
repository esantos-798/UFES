% ==========================================================
%  MODELAGEM DE RESPOSTA AO IMPULSO
%  - Underwater Wireless Optical Communication (UWOC) -
% ==========================================================
%  Universidade Federal do Espírito Santo
%  Programa de Pós-Graduação em Engenharia Elétrica
%
%  Cristiano Tavares                 Jair Adriano Silva Lima
%                                                 Julho/2026
% ==========================================================

clear; clc; close all;

% ==========================================================
% Parâmetros de entrada
% ==========================================================
num_fotons = .5e6;          % Número de fótons
L = 50;                     % Distância
a = 0.088;                  % Absorção (Coastal)
b = 0.216;                  % Espalhamento (Coastal)
v = 2.25e8;                 % Velocidade da luz na água
g = 0.9470;                 % Fator HG (Henyey–Greenstein)
raio_detector = 1;          % Receptor [m]

janela_ns = 45;             % Janela de observação
passo_s = 0.1e-9;           % Passo do histograma de Monte Carlo

c = a + b;                  % Coeficiente de extinção

% ==========================================================
% Simulação de Monte Carlo / histograma
% ==========================================================

% Lançamento dos fótons
t_chegada = [];
WB = waitbar(0, 'Executando...','Name', 'Simulação Monte Carlo');       %WaitBar
for count = 1:num_fotons       
    pos = [0, 0, 0];        %vetor posição inicial
    dir = [0, 0, 1];        %vetor direção
    dist_total = 0;

    while pos(1) < L && dist_total < 5*L
        dist_passo = -log(rand()) / c;
        pos = pos + dir * dist_passo;
        dist_total = dist_total + dist_passo;

        r = rand();
        cos_theta = (1/(2*g)) * (1 + g^2 - ((1-g^2)/(1-g+2*g*r))^2);
        sin_theta = sqrt(1 - cos_theta^2);
        phi = 2*pi*rand();

        nova_dir = [cos_theta, sin_theta*cos(phi), sin_theta*sin(phi)];
        %nova_dir = [sin_theta*cos(phi), sin_theta*sin(phi),
        %nova_dir = [(cos_theta/abs(cos_theta))*cos_theta, sin_theta*cos(phi), sin_theta*sin(phi)];

        dir = nova_dir / norm(nova_dir);
    end

    % Verifica se passou do plano do receptor
    if pos(1) >= L
        % Calcula onde o fóton furou o plano X
        retrocesso = (pos(1) - L) / dir(1);
        y_hit = pos(2) - dir(2) * retrocesso;
        z_hit = pos(3) - dir(3) * retrocesso;

        % Só conta se acertou a lente do detector físico!
        if (y_hit^2 + z_hit^2) <= raio_detector^2
            t_chegada = [t_chegada, dist_total / v];
        end
    end

    %fprintf('%d\n',count);
    if mod(count,(num_fotons/20)) == 0
        waitbar(count/num_fotons, WB, sprintf('%.1f%% concluído', 100*count/num_fotons));
    end
end


% Tratamento dos dados de saída
t0 = L / v;
    
t_limite = t0 + (janela_ns * 1e-9);
t_chegada_valido = t_chegada(t_chegada <= t_limite);

% Criação das caixas de tempo do histograma
t_bins = t0 : passo_s : t_limite;
t_centro = t_bins(1:end-1) + passo_s/2;



contagens = histcounts(t_chegada_valido, t_bins);
h_mc = contagens / num_fotons; 
h_max=max(h_mc);

if h_max == 0
    error('NENHUM fóton atingiu o receptor!');
end

% Conversão do centro do tempo para nanosegundos (t - t0)
t0 = min(t_bins);
t_data_ns = (t_centro - t0) * 1e9;
h_mc_norm = h_mc / h_max;

% ==========================================================
% Modelo Double Gamma Functions - DGF
% ==========================================================

% Função anônima DGF
dgf_norm = @(C, t) (t >= 0) .* (C(1).*t.*exp(-C(2).*t) + C(3).*t.*exp(-C(4).*t));

C0 = [1, 0.5, 1, 0.5];      % valor inicial de C
lb = [0, 0, 0, 0];          % limite inferior
ub = [10, 10, 10, 10];      % limite superior

% Busca/Otimização dos valores dos coeficientes Cs
options = optimoptions('lsqcurvefit', ...
    'Algorithm', 'trust-region-reflective', ...
    'Display', 'iter', ...
    'FunctionTolerance', 1e-12);


C_finais = lsqcurvefit(dgf_norm, C0, t_data_ns, h_mc_norm, lb, ub, options);

% Desnormalização de C
escala_tempo = 1e9;
C_original_1 = C_finais(1) * h_max * escala_tempo;
C_original_2 = C_finais(2) * escala_tempo;
C_original_3 = C_finais(3) * h_max * escala_tempo;
C_original_4 = C_finais(4) * escala_tempo;


fprintf('\n=== COEFICIENTES REAIS DA GAMA DUPLA ===\n');
fprintf('C1 = %.4e\nC2 = %.4e\nC3 = %.4e\nC4 = %.4e\n', ...
    C_original_1, C_original_2, C_original_3, C_original_4);


% ==========================================================
% Gráfico final
% ==========================================================
close(WB)

delta_t_segundos = t_centro - t0;
h_verificacao = (C_original_1 .* delta_t_segundos .* exp(-C_original_2 .* delta_t_segundos)) + ...
    (C_original_3 .* delta_t_segundos .* exp(-C_original_4 .* delta_t_segundos));

figure('Color', 'w', 'Name', 'Resposta ao Impulso');

bar(delta_t_segundos * 1e9, h_mc,'DisplayName', 'Monte Carlo');
hold on;
%plot(delta_t_segundos * 1e9, h_mc, 'ko', 'DisplayName', 'Monte Carlo', 'MarkerSize', 6);
plot(delta_t_segundos * 1e9, h_verificacao, 'r-','LineWidth', 2.5,'DisplayName', 'Modelo Gama Dupla');


xlabel('Tempo após t_0 (ns)', 'FontWeight', 'bold'); 
ylabel('Resposta ao Impulso do Canal [h(t)]', 'FontWeight', 'bold');
title('Ajuste do Modelo de Canal UWOC');
legend; 
grid on;
