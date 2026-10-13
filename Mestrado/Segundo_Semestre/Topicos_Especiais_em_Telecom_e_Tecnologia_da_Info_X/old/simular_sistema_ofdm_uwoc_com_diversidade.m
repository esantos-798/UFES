% =========================================================================
% SIMULAÇÃO TRANSMISSOR/RECEPTOR OFDM EM CANAL UWOC COM DIVERSIDADE
% Alinhado com as formulações de Turbulência (Artigo 1) e Diversidade (Artigo 3)
% Compatível com MATLAB e GNU Octave
% =========================================================================
clear; clc; close all;
warning('off', 'Octave:graphics-toolkit');
warning('off', 'Octave:legacy-graphics');

%% 1. Configurações Globais da Simulação
N_subcarriers = 64;             % Número de subportadoras OFDM
N_symbols = 500;                % Quantidade de símbolos OFDM para estatística de BER
M = 4;                          % QPSK (4-QAM)
bits_per_symbol = log2(M);
sigma_bolhas = 0.35;            % Severidade da turbulência/bolhas (Artigo 1)
vetor_EbN0_dB = 0:2:16;         % Faixa de simulação de relação sinal-ruído

% Configuração de Diversidade Espacial (Artigo 3)
% Compararemos 1 receptor (Sem Diversidade) vs N receptores (Com Diversidade)
N_receptores = 4; 

% Vetores para armazenar as taxas de erro (BER)
BER_sem_diversidade = zeros(size(vetor_EbN0_dB));
BER_com_diversidade = zeros(size(vetor_EbN0_dB));

% Mapeamento QPSK Constelação (Unit Average Power)
constelacao = [1+1i, -1+1i, -1-1i, 1-1i] / sqrt(2);

fprintf('Iniciando simulação Monte Carlo para o canal UWOC...\n');

%% 2. Loop de Varredura de SNR (Eb/N0)
for idx_snr = 1:length(vetor_EbN0_dB)
    EbN0_dB = vetor_EbN0_dB(idx_snr);
    
    % Conversão de Eb/N0 para variância do ruído complexo por subportadora
    EbN0 = 10^(EbN0_dB/10);
    EsN0 = EbN0 * bits_per_symbol;
    % Variância do ruído para cada componente (I e Q)
    sigma_ruido = sqrt(1 / (2 * EsN0)); 
    
    erros_antena1 = 0;
    erros_mrc = 0;
    total_bits = 0;
    
    for sym = 1:N_symbols
        %% 3. Transmissor OFDM
        % Geração de bits aleatórios
        tx_bits = randi([0 1], N_subcarriers * bits_per_symbol, 1);
        
        % Agrupamento de bits em símbolos QPSK (Mapeamento Manual)
        tx_simbolos_idx = tx_bits(1:2:end)*2 + tx_bits(2:2:end);
        % Ajustando índices de 0:3 para 1:4 do MATLAB
        tx_simbolos = constelacao(tx_simbolos_idx + 1).'; 
        
        % Modulação OFDM (IFFT)
        sinal_tx = ifft(tx_simbolos) * sqrt(N_subcarriers);
        
        %% 4. Canal de Comunicação Ótica Subaquática (UWOC)
        sinais_rx = zeros(N_subcarriers, N_receptores);
        fading_canais = zeros(N_subcarriers, N_receptores);
        
        for r = 1:N_receptores
            % Modelo Log-Normal para flutuação de envelope induzida por bolhas (Artigo 1)
            % Média nula no expoente ajustada para manter a energia média unitária
            fading_bolhas = exp(sigma_bolhas * randn(N_subcarriers, 1) - (sigma_bolhas^2)/2);
            fading_canais(:, r) = fading_bolhas;
            
            % Ruído Gaussiano Complexo de Fundo
            ruido = (randn(N_subcarriers, 1) + 1i*randn(N_subcarriers, 1)) * sigma_ruido;
            
            % Sinal recebido no fotodetector r
            sinais_rx(:, r) = (sinal_tx .* fading_bolhas) + ruido;
        end
        
        %% 5. Receptor com Processamento de Sinais e Diversidade (Artigo 3)
        % --- CASO A: Sem Diversidade (Apenas Receptor 1) ---
        sinal_rx_antena1 = sinais_rx(:, 1);
        % Equalização simples baseada no ganho de canal conhecido
        espectro_rx_antena1 = fft(sinal_rx_antena1) / sqrt(N_subcarriers);
        simbolos_eq_antena1 = espectro_rx_antena1 ./ fading_canais(:, 1);
        
        % --- CASO B: Com Diversidade Espacial via Combinador MRC ---
        % Maximal Ratio Combining: Pondera cada canal pelo seu ganho conjugado
        sinal_mrc_combinado = zeros(N_subcarriers, 1);
        denominador_mrc = zeros(N_subcarriers, 1);
        
        for r = 1:N_receptores
            sinal_mrc_combinado = sinal_mrc_combinado + (sinais_rx(:, r) .* fading_canais(:, r));
            denominador_mrc = denominador_mrc + (fading_canais(:, r).^2);
        end
        sinal_mrc_equalizado = sinal_mrc_combinado ./ denominador_mrc;
        
        % Demodulação OFDM do sinal combinado via FFT
        espectro_rx_mrc = fft(sinal_mrc_equalizado) / sqrt(N_subcarriers);
        
        %% 6. Demapeamento e Decisão de Bits (Máxima Verossimilhança)
        rx_bits_antena1 = zeros(size(tx_bits));
        rx_bits_mrc = zeros(size(tx_bits));
        
        for k = 1:N_subcarriers
            % Distâncias Euclidianas para Antena 1
            [~, idx_est_a1] = min(abs(simbolos_eq_antena1(k) - constelacao));
            % Distâncias Euclidianas para MRC
            [~, idx_est_mrc] = min(abs(espectro_rx_mrc(k) - constelacao));
            
            % Conversão de volta para bits binários
            bit_map = [0 0; 0 1; 1 0; 1 1]; % Mapeamento inverso
            rx_bits_antena1(2*k-1 : 2*k) = bit_map(idx_est_a1, :);
            rx_bits_mrc(2*k-1 : 2*k)     = bit_map(idx_est_mrc, :);
        end
        
        % Acumulação de erros
        erros_antena1 = erros_antena1 + sum(tx_bits ~= rx_bits_antena1);
        erros_mrc = erros_mrc + sum(tx_bits ~= rx_bits_mrc);
        total_bits = total_bits + length(tx_bits);
    end
    
    % Cálculo final da BER para este ponto de SNR
    BER_sem_diversidade(idx_snr) = erros_antena1 / total_bits;
    BER_com_diversidade(idx_snr) = erros_mrc / total_bits;
    
    fprintf('Eb/N0 = %d dB concluído. BER Sem Div: %.4e | BER Com Div: %.4e\n', ...
        EbN0_dB, BER_sem_diversidade(idx_snr), BER_com_diversidade(idx_snr));
end
% Substitui valores de BER igual a 0 por um limite inferior adequado para o log plot
% Isso evita que o semilogy tente calcular log10(0) a partir de 4 dB
BER_com_diversidade(BER_com_diversidade == 0) = 1e-6; 

% Força o uso do gnuplot se estiver no Octave, evitando quebras pelo QT
if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    % Tenta gnuplot para máxima compatibilidade em CLI, se falhar mantém o padrão
    try
        graphics_toolkit('gnuplot');
    catch
        % Se gnuplot não estiver disponível, deixa o Octave decidir
    end
end
%% 7. Geração do Gráfico de Desempenho Analítico
figure('Color', [1 1 1]);
semilogy(vetor_EbN0_dB, BER_sem_diversidade, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
hold on;
semilogy(vetor_EbN0_dB, BER_com_diversidade, 'b-s', 'LineWidth', 2, 'MarkerFaceColor', 'b');
grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Taxa de Erro de Bit (BER)', 'FontSize', 11, 'FontWeight', 'bold');
title({'Desempenho do Transceptor OFDM em Canal Ótico Subaquático'; ...
       ['Efeito de Turbulência por Bolhas (\sigma = ' num2str(sigma_bolhas) ') vs Diversidade Espacial']}, ...
       'FontSize', 12, 'FontWeight', 'bold');
legend('Sinal sem Diversidade (1 RX)', ['Diversidade Espacial MRC (' num2str(N_receptores) ' RXs)'], ...
       'Location', 'SouthWest');
set(gca, 'FontSize', 10);