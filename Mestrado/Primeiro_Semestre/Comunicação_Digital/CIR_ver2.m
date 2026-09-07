
clear; clc; close all;



% --- 1. CHAMADA DA SIMULAÇÃO (Física) ---
[h_mc_bruto, t_bins] = simular_mc_uwoc_HG();


t_data_ns = (t_bins(1:end-1) - min(t_bins)) * 1e9; % Tempo em ns normalizado
h_mc_norm = h_mc_bruto / max(h_mc_bruto);          % Amplitude normalizada (0 a 1)

% --- 2. MODELO MATEMÁTICO ---
% C1 e C3 agora operam na escala 0-1, C2 e C4 na escala de decaimento por ns
dgf_norm = @(C, t) (t >= 0) .* (C(1).*t.*exp(-C(2).*t) + C(3).*t.*exp(-C(4).*t));

% --- 3. CHUTE INICIAL "ESPERTO" (Baseado na física da curva) ---
% C2 e C4 são taxas de decaimento. Se o pulso dura 5ns, um chute inicial 
% de 0.5 a 1.0 é um ponto de partida seguro e estável.
C0 = [0.1, 0.5, 0.1, 0.5]; 

% --- 4. OTIMIZAÇÃO ROBUSTA ---
options = optimoptions('lsqcurvefit', ...
    'Algorithm', 'trust-region-reflective', ...
    'Display', 'iter', ...
    'FunctionTolerance', 1e-12);

% Limites para garantir estabilidade (C2 e C4 não podem ser negativos!)
lb = [0, 0, 0, 0];
ub = [10, 10, 10, 10]; 

C_finais = lsqcurvefit(dgf_norm, C0, t_data_ns, h_mc_norm, lb, ub, options);

% --- 5. PLOTAGEM DE RESULTADOS ---
figure('Color', 'w');
plot(t_data_ns, h_mc_norm, 'k.', 'MarkerSize', 10, 'DisplayName', 'Monte Carlo (Simulado)');
hold on;
plot(t_data_ns, dgf_norm(C_finais, t_data_ns), 'r-', 'LineWidth', 2.5, 'DisplayName', 'Ajuste Gama Dupla');
xlabel('Tempo [ns]'); ylabel('Intensidade Normalizada');
legend; grid on;

fprintf('\nCoeficientes encontrados para o modelo:\n');
disp(C_finais);
