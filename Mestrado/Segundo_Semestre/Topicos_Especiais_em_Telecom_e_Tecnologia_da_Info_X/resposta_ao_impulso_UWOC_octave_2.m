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
%raio_detector = 1;         % Receptor [m]
raio_detector = 0.25;       % Receptor [m] (Tang)
angulo_divergencia = 10;    % Divergência total da fonte [graus] (Tang)
%a divergência de 10° foi assumida como o ângulo total correspondente ao semiângulo de meia potência do modelo Lambertiano generalizado
W_min = 1e-6;               % Peso mínimo considerado na degradação


janela_ns = 16;             % Janela de observação
passo_s = 0.1e-9;           % Passo do histograma de Monte Carlo

% ----------------------------------------------------------
% Cálculos iniciais 
% ----------------------------------------------------------
c = a + b;                  % Coeficiente de extinção

theta_half = deg2rad(angulo_divergencia/2);

% Ordem Lambertiana
m_led = -log(2) / log(cos(theta_half));

% ==========================================================
% Simulação de Monte Carlo / histograma
% ==========================================================

% Lançamento dos fótons
t_chegada = [];
peso_chegada = [];

% WaitBar
WB = waitbar(0, 'Executando...','Name', 'Simulação Monte Carlo');

for count = 1:num_fotons       
    pos = [0, 0, 0];        %vetor posição inicial 

    % ======================================================
    % Direção inicial do fóton     
    % LED - modelo Lambertiano generalizado de Cox
    % ======================================================
        
    % Número aleatório uniforme
    R_theta = rand();
    
    % Ângulo polar segundo a CDF da fonte Lambertiana
    theta0 = acos((1 - R_theta)^(1/(m_led + 1)));
    
    % Azimute uniforme
    phi0 = 2*pi*rand();
    
    % Direção inicial - eixo principal da fonte em +X
    dir = [cos(theta0), sin(theta0)*cos(phi0), sin(theta0)*sin(phi0)];
    dir = dir / norm(dir);   
    
    dist_total = 0;

    W = 1;  %Peso do fóton inicial
   
    while pos(1) < L && dist_total < 5*L
    
        % ======================================================
        % 1. Sorteia a distância até a próxima interação
        % ======================================================
        dist_passo = -log(rand()) / c;
    
    
        % ======================================================
        % 2. Verifica se o fóton encontra o plano receptor
        % ======================================================
    
        if dir(1) > 0
    
            % Distância necessária, na direção atual, para alcançar o plano x = L
            dist_ate_rx = (L - pos(1)) / dir(1);
    
            % Se o plano receptor estiver dentro do passo sorteado, o fóton chega ao receptor antes da próxima interação
            if dist_ate_rx >= 0 && dist_ate_rx <= dist_passo
    
                % Posição EXATA onde o fóton cruza o plano x = L
                pos_hit = pos + dir * dist_ate_rx;
    
                y_hit = pos_hit(2);
                z_hit = pos_hit(3);
    
                % Verifica se o fóton está dentro da área do detector
                if (y_hit^2 + z_hit^2) <= raio_detector^2
    
                    % Distância total REAL percorrida até o receptor
                    dist_chegada = dist_total + dist_ate_rx;
    
                    % Tempo de chegada
                    t_chegada = [t_chegada, dist_chegada / v];
                    peso_chegada = [peso_chegada, W];
    
                end
    
                % O fóton atingiu o plano receptor.
                % Não deve continuar sendo propagado.
                break;
    
            end
        end
    
    
        % ======================================================
        % 3. Se NÃO atingiu o receptor, percorre todo o passo
        % ======================================================
    
        pos = pos + dir * dist_passo;
        dist_total = dist_total + dist_passo;

        % ======================================================
        % Absorção: atualização de degradação do peso do fóton
        % ======================================================
        
        W = W * (1 - a/c);          %Fórmula Tang
        
        % Interrompe o rastreamento se o peso ficar muito baixo
        if W < W_min
            break;
        end
    
    
        % ======================================================
        % 4. Espalhamento  Henyey-Greenstein
        % ======================================================
    
        r = rand();
    
        cos_theta = (1/(2*g)) * (1 + g^2 - ((1-g^2)/(1-g+2*g*r))^2);

        % Truncamento para cos_theta muito pequeno
        cos_theta = max(-1, min(1, cos_theta));
    
        sin_theta = sqrt(1 - cos_theta^2);
    
        phi = 2*pi*rand();
    
        
        % ======================================================
        % Nova direção RELATIVA à direção atual do fóton
        % ======================================================
        
        % Direção atual normalizada
        u = dir / norm(dir);
        
        % Cria um vetor perpendicular à direção atual
        if abs(u(3)) < 0.999
            e1 = cross([0 0 1], u);
        else
            e1 = cross([0 1 0], u);
        end
        
        e1 = e1 / norm(e1);
        
        % Segundo vetor perpendicular
        e2 = cross(u, e1);
        
        % Aplica theta e phi em relação à direção anterior
        nova_dir = cos_theta * u + sin_theta * cos(phi) * e1 + sin_theta * sin(phi) * e2;
        
        % Normalização por segurança
        dir = nova_dir / norm(nova_dir);
    end

    %fprintf('%d\n',count);
    if mod(count,(num_fotons/100)) == 0
        waitbar(count/num_fotons, WB, sprintf('%.1f%% concluído', 100*count/num_fotons));
    end
end


% Tratamento dos dados de saída
t0 = L / v;
    
t_limite = t0 + (janela_ns * 1e-9);

%Verifica se os tempos de chegada são válidos
validos = false(size(t_chegada));
for i = 1:length(t_chegada)

    if t_chegada(i) <= t_limite
        validos(i) = true;
    end

end
t_chegada_valido = t_chegada(validos);          %fica apenas com os válidos
peso_chegada_valido = peso_chegada(validos);


% Criação das caixas de tempo do histograma
t_bins = t0 : passo_s : t_limite;
t_centro = t_bins(1:end-1) + passo_s/2;


% ======================================================
% Montagem do histograma
% ======================================================
% Identifica em qual bin temporal cada fóton caiu
[~,~,bin] = histcounts(t_chegada_valido, t_bins);

% Ignora valores fora dos bins
idx = bin > 0;

% Soma o PESO dos fótons em cada intervalo de tempo
contagens = accumarray(bin(idx).', peso_chegada_valido(idx).', [length(t_bins)-1, 1], @sum, 0).';

% Resposta ao impulso
h_mc = contagens / num_fotons;


h_max=max(h_mc);

if h_max == 0
    error('NENHUM fóton atingiu o receptor!');
end

% Conversão do centro do tempo para nanosegundos (t - t0)
t0 = min(t_bins);
t_data_ns = (t_centro - t0) * 1e9;
h_mc_norm = h_mc / h_max;


% % ==========================================================
% % Delay spread a -20 dB (Bharathi)
% % ==========================================================
% 
% limiar_20dB = 0.01;
% num_bins_confirmacao = 5;
% 
% [~, indice_pico] = max(h_mc_norm);
% 
% indice_queda = [];
% 
% for i = indice_pico:(length(h_mc_norm) - num_bins_confirmacao + 1)
% 
%     abaixo = true;
% 
%     for j = 0:(num_bins_confirmacao - 1)
% 
%         if h_mc_norm(i+j) >= limiar_20dB
%             abaixo = false;
%             break;
%         end
% 
%     end
% 
%     if abaixo
%         indice_queda = i;
%         break;
%     end
% 
% end
% 
% if ~isempty(indice_queda)
% 
%     delay_spread_20dB = ...
%         t_data_ns(indice_queda) - t_data_ns(indice_pico);
% 
%     fprintf('Tempo do pico: %.2f ns\n', ...
%         t_data_ns(indice_pico));
% 
%     fprintf('Delay spread (-20 dB): %.2f ns\n', ...
%         delay_spread_20dB);
% 
% else
% 
%     fprintf('A resposta não caiu abaixo de -20 dB na janela.\n');
% 
% end


% ==========================================================
% Modelo Double Gamma Functions - DGF
% ==========================================================

% Função anônima DGF
dgf_norm = @(C, t) (t >= 0) .* (C(1).*t.*exp(-C(2).*t) + C(3).*t.*exp(-C(4).*t));

C0 = [1, 0.5, 1, 0.5];      % valor inicial de C
lb = [0, 0, 0, 0];          % limite inferior
ub = [10, 10, 10, 10];      % limite superior

% Busca/Otimização dos valores dos coeficientes Cs
options = optimoptions('lsqcurvefit', 'Algorithm', 'trust-region-reflective', 'Display', 'iter', 'FunctionTolerance', 1e-12);

C_finais = lsqcurvefit(dgf_norm, C0, t_data_ns, h_mc_norm, lb, ub, options);


% ==========================================================
% Avaliação do ajuste - R² e RMSE
% ==========================================================

% Resposta prevista pela função Double Gamma
h_dgf_norm = dgf_norm(C_finais, t_data_ns);

% ----------------------------------------------------------
% RMSE
% ----------------------------------------------------------

erro = h_mc_norm - h_dgf_norm;

RMSE = sqrt(mean(erro.^2));


% ----------------------------------------------------------
% Coeficiente de determinação R²
% ----------------------------------------------------------

% Soma dos quadrados dos resíduos
SS_res = sum((h_mc_norm - h_dgf_norm).^2);

% Soma total dos quadrados
SS_tot = sum((h_mc_norm - mean(h_mc_norm)).^2);

R2 = 1 - (SS_res / SS_tot);


% ==========================================================
% Exibição dos resultados
% ==========================================================

fprintf('\n=== INFORMAÇÕES DA SIMULAÇÃO ===\n');
fprintf('Fótons emitidos: %d\n', num_fotons);
fprintf('Fótons detectados: %d\n', length(t_chegada));
fprintf('Soma dos pesos recebidos: %.6e\n', sum(peso_chegada));
fprintf('Ordem Lambertiana m = %.2f\n',m_led);


fprintf('\n=== QUALIDADE DO AJUSTE DOUBLE GAMMA ===\n');
fprintf('R²   = %.6f\n', R2);
fprintf('RMSE = %.6f\n', RMSE);


fprintf('\n=== COEFICIENTES NORMALIZADOS DA GAMA DUPLA ===\n');
fprintf('C1 = %.6f\n', C_finais(1));
fprintf('C2 = %.6f\n', C_finais(2));
fprintf('C3 = %.6f\n', C_finais(3));
fprintf('C4 = %.6f\n', C_finais(4));

% Desnormalização de C
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
