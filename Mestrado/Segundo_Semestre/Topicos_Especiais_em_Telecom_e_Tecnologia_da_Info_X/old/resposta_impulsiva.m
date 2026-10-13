% =========================================================================
% Modelagem da Resposta Impulsiva de Canal (CIR) para Sistemas UWOC
% Baseado no Modelo de Função Double Gamma (DGF) dos Artigos de Sahu et al.
% Compatível com MATLAB e GNU Octave
% =========================================================================

clear; clc; close all;

% Definição do vetor de tempo normalizado delta_t (em nanossegundos - ns)
% Representa o tempo de atraso após a chegada do primeiro fotão balístico.
dt = 0.01; % Passo de amostragem em ns
t_max = 5; % Tempo máximo de simulação em ns
delta_t = 0:dt:t_max;

%% 1. PARÂMETROS DE AJUSTE (Extraídos das Tabelas do Artigo)

% --- ÁGUA COSTEIRA (Coastal Water) ---
% Tabela 3 do Artigo (Para AL = 4, 8, 12)
coastal_params = [
%   AL    C1      C2      C3      C4
    4,  0.4180,  11.060,  9280,  68.68;
    8,  0.3958,   8.811,  1589,  51.35;
   12,  0.3141,   5.884,  696.7,  43.79
];

% --- ÁGUA DE PORTO (Harbor Water - Turbid) ---
% Tabela 4 do Artigo (Para AL = 4, 8, 12)
harbor_params = [
%   AL    C1      C2      C3      C4
    4,  1.1430,  7.196,  109.30,  27.79;
    8,  0.8373,  4.831,   45.00,  22.44;
   12,  0.4466,  2.953,   12.23,  15.91
];

%% 2. FUNÇÃO MODELO (Double Gamma Function)
% h(t) = C1 * dt * exp(-C2 * dt) + C3 * dt * exp(-C4 * dt)
dgf_model = @(t, C) C(2)*t.*exp(-C(3)*t) + C(4)*t.*exp(-C(5)*t);

%% 3. SIMULAÇÃO E GERAÇÃO DOS GRÁFICOS

figure('Position', [100, 100, 1000, 500]);

% --- Gráfico Esquerdo: Água Costeira ---
subplot(1, 2, 1);
hold on;
grid on;
colors_coastal = {'b', 'g', 'r'};

for i = 1:size(coastal_params, 1)
    C = coastal_params(i, :);
    AL_val = C(1);
    
    % Calcula a resposta impulsiva
    h_t = dgf_model(delta_t, C);
    
    % Normalização da potência recebida (opcional, para visualização igual ao artigo)
    h_t_norm = h_t / max(h_t);
    
    plot(delta_t, h_t_norm, colors_coastal{i}, 'LineWidth', 2, ...
         'DisplayName', sprintf('Costeira (AL = %d)', AL_val));
end

title('Resposta Impulsiva - Água Costeira');
xlabel('Tempo de Atraso Normalizado \Delta t (ns)');
ylabel('Potência Recebida Normalizada');
legend('show', 'Location', 'northeast');
xlim([0 t_max]);
ylim([0 1.1]);

% --- Gráfico Direito: Água de Porto ---
subplot(1, 2, 2);
hold on;
grid on;
colors_harbor = {'m', 'c', 'k'};

for i = 1:size(harbor_params, 1)
    C = harbor_params(i, :);
    AL_val = C(1);
    
    % Calcula a resposta impulsiva
    h_t = dgf_model(delta_t, C);
    
    % Normalização
    h_t_norm = h_t / max(h_t);
    
    plot(delta_t, h_t_norm, colors_harbor{i}, 'LineWidth', 2, ...
         'DisplayName', sprintf('Porto (AL = %d)', AL_val));
end

title('Resposta Impulsiva - Água de Porto (Turbid)');
xlabel('Tempo de Atraso Normalizado \Delta t (ns)');
ylabel('Potência Recebida Normalizada');
legend('show', 'Location', 'northeast');
xlim([0 t_max]);
ylim([0 1.1]);

% Ajuste estético geral
sgtitle('Modelagem Empírica da CIR via Função Double Gamma (DGF)');