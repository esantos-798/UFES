clear; clc; close all;

% Se estiver no Octave, carrega o pacote de otimização necessário
if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    pkg load optim;
end

% --- 1. CHAMADA DA SIMULAÇÃO (Física) ---
[h_mc_bruto, t_bins] = simular_mc_uwoc_HG_1();

t_data_ns = (t_bins(1:end-1) - min(t_bins)) * 1e9; % Tempo em ns normalizado
h_mc_norm = h_mc_bruto / max(h_mc_bruto);          % Amplitude normalizada (0 a 1)

% --- 2. MODELO MATEMÁTICO ---
dgf_norm = @(C, t) (t >= 0) .* (C(1).*t.*exp(-C(2).*t) + C(3).*t.*exp(-C(4).*t));

% --- 3. CHUTE INICIAL "ESPERTO" ---
C0 = [0.1, 0.5, 0.1, 0.5]; 

% --- 4. OTIMIZAÇÃO ROBUSTA (Adaptada para MATLAB e Octave) ---
lb = [0, 0, 0, 0];
ub = [10, 10, 10, 10]; 

if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    % Configuração para o Octave usando optimset
    options = optimset('Display', 'iter', 'TolX', 1e-12);
    % No Octave, lsqcurvefit aceita os limites dentro do optimset ou direto na função dependendo da versão
    C_finais = lsqcurvefit(dgf_norm, C0, t_data_ns, h_mc_norm, lb, ub, options);
else
    % Configuração para o MATLAB usando optimoptions
    options = optimoptions('lsqcurvefit', ...
        'Algorithm', 'trust-region-reflective', ...
        'Display', 'iter', ...
        'FunctionTolerance', 1e-12);
    C_finais = lsqcurvefit(dgf_norm, C0, t_data_ns, h_mc_norm, lb, ub, options);
end

% --- 5. PLOTAGEM DE RESULTADOS ---
figure('Color', 'w');
plot(t_data_ns, h_mc_norm, 'k.', 'MarkerSize', 10, 'DisplayName', 'Monte Carlo (Simulado)');
hold on;
plot(t_data_ns, dgf_norm(C_finais, t_data_ns), 'r-', 'LineWidth', 2.5, 'DisplayName', 'Ajuste Gama Dupla');
xlabel('Tempo [ns]'); ylabel('Intensidade Normalizada');
legend('show'); % 'show' garante compatibilidade estrita no Octave
grid on;

fprintf('\nCoeficientes encontrados para o modelo:\n');
disp(C_finais);

disp('Pressione ENTER para fechar...');
pause();