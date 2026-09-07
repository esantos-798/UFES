% =========================================================================
% SIMULAÇÃO TRANSMISSOR/RECEPTOR OFDM EM CANAL UWOC COM DIVERSIDADE
% Foco: Turbulência (Artigo 1), Diversidade (Artigo 3) e Efeito Passa-Baixas
% Compatível com MATLAB e GNU Octave
% =========================================================================
clear; clc; close all;

% Desativa os avisos de interface gráfica do Octave no terminal
warning('off', 'Octave:graphics-toolkit');
warning('off', 'Octave:legacy-graphics');

%% 1. Configurações Globais da Simulação
N_subcarriers = 64;             % Número de subportadoras OFDM
N_symbols = 500;                % Quantidade de símbolos OFDM para estatística
M = 4;                          % QPSK (4-QAM)
bits_per_symbol = log2(M);
sigma_bolhas = 0.35;            % Severidade da turbulência/bolhas (Artigo 1)
vetor_EbN0_dB = 0:2:16;         % Faixa de simulação de SNR

% Configuração de Diversidade Espacial (Artigo 3)
N_receptores = 4; 

% Vetores para armazenar as taxas de erro (BER)
BER_sem_diversidade = zeros(size(vetor_EbN0_dB));
BER_com_diversidade = zeros(size(vetor_EbN0_dB));

% Mapeamento QPSK Constelação (Unit Average Power)
constelacao = [1+1i, -1+1i, -1-1i, 1-1i] / sqrt(2);

%% 2. Definição do Efeito Passa-Baixas do Canal Ótico
% Vamos criar uma atenuação exponencial que simula a resposta passa-baixas.
% Subportadoras centrais (baixa frequência) passam bem, subportadoras de alta frequência sofrem atenuação.
frequencias_relativas = (0:N_subcarriers-1) / N_subcarriers;
% Canal passa-baixas: subportadoras mais altas sofrem maior atenuação (ex: decaimento de 3dB ou mais)
H_passa_baixas = exp(-1.2 * frequencias_relativas).'; 

fprintf('Iniciando simulação Monte Carlo para o canal UWOC com Canal Passa-Baixas...\n');

%% 3. Loop de Varredura de SNR (Eb/N0)
for idx_snr = 1:length(vetor_EbN0_dB)
    EbN0_dB = vetor_EbN0_dB(idx_snr);
    
    EbN0 = 10^(EbN0_dB/10);
    EsN0 = EbN0 * bits_per_symbol;
    sigma_ruido = sqrt(1 / (2 * EsN0)); 
    
    erros_antena1 = 0;
    erros_mrc = 0;
    total_bits = 0;
    
    for sym = 1:N_symbols
        %% 4. Transmissor OFDM
        tx_bits = randi([0 1], N_subcarriers * bits_per_symbol, 1);
        tx_simbolos_idx = tx_bits(1:2:end)*2 + tx_bits(2:2:end);
        tx_simbolos = constelacao(tx_simbolos_idx + 1).'; 
        
        % Modulação OFDM (Espectro)
        % Aplicamos o efeito passa-baixas diretamente no domínio da frequência (na característica do canal)
        tx_espectro_filtrado = tx_simbolos .* H_passa_baixas;
        
        % Conversão para o domínio do tempo (IFFT)
        sinal_tx = ifft(tx_espectro_filtrado) * sqrt(N_subcarriers);
        
        %% 5. Canal de Comunicação Ótica Subaquática (UWOC)
        sinais_rx = zeros(N_subcarriers, N_receptores);
        fading_canais = zeros(N_subcarriers, N_receptores);
        
        for r = 1:N_receptores
            % Modelo Log-Normal para flutuação por bolhas (Artigo 1)
            fading_bolhas = exp(sigma_bolhas * randn(N_subcarriers, 1) - (sigma_bolhas^2)/2);
            fading_canais(:, r) = fading_bolhas;
            
            % Ruído Gaussiano Complexo de Fundo (AWGN no Fotodetector)
            ruido = (randn(N_subcarriers, 1) + 1i*randn(N_subcarriers, 1)) * sigma_ruido;
            
            % Sinal recebido no fotodetector r
            sinais_rx(:, r) = (sinal_tx .* fading_bolhas) + ruido;
        end
        
        %% 6. Receptor com Processamento de Sinais e Diversidade (Artigo 3)
        % --- CASO A: Sem Diversidade (Apenas Receptor 1) ---
        sinal_rx_antena1 = sinais_rx(:, 1);
        espectro_rx_antena1 = fft(sinal_rx_antena1) / sqrt(N_subcarriers);
        
        % Equalização Zero-Forcing (ZF) que tenta corrigir o fading E o efeito passa-baixas
        simbolos_eq_antena1 = espectro_rx_antena1 ./ (fading_canais(:, 1) .* H_passa_baixas);
        
        % --- CASO B: Com Diversidade Espacial via Combinador MRC ---
        sinal_mrc_combinado = zeros(N_subcarriers, 1);
        denominador_mrc = zeros(N_subcarriers, 1);
        
        for r = 1:N_receptores
            sinal_mrc_combinado = sinal_mrc_combinado + (sinais_rx(:, r) .* fading_canais(:, r));
            denominador_mrc = denominador_mrc + (fading_canais(:, r).^2);
        end
        sinal_mrc_equalizado = sinal_mrc_combinado ./ denominador_mrc;
        
        % Demodulação OFDM do sinal combinado via FFT e compensação do filtro passa-baixas
        espectro_rx_mrc = fft(sinal_mrc_equalizado) / sqrt(N_subcarriers);
        simbolos_eq_mrc = espectro_rx_mrc ./ H_passa_baixas;
        
        %% 7. Demapeamento e Decisão de Bits (Máxima Verossimilhança)
        rx_bits_antena1 = zeros(size(tx_bits));
        rx_bits_mrc = zeros(size(tx_bits));
        
        for k = 1:N_subcarriers
            [~, idx_est_a1] = min(abs(simbolos_eq_antena1(k) - constelacao));
            [~, idx_est_mrc] = min(abs(simbolos_eq_mrc(k) - constelacao));
            
            bit_map = [0 0; 0 1; 1 0; 1 1]; 
            rx_bits_antena1(2*k-1 : 2*k) = bit_map(idx_est_a1, :);
            rx_bits_mrc(2*k-1 : 2*k)     = bit_map(idx_est_mrc, :);
        end
        
        erros_antena1 = erros_antena1 + sum(tx_bits ~= rx_bits_antena1);
        erros_mrc = erros_mrc + sum(tx_bits ~= rx_bits_mrc);
        total_bits = total_bits + length(tx_bits);
    end
    
    BER_sem_diversidade(idx_snr) = erros_antena1 / total_bits;
    BER_com_diversidade(idx_snr) = erros_mrc / total_bits;
    
    fprintf('Eb/N0 = %d dB concluído. BER Sem Div: %.4e | BER Com Div: %.4e\n', ...
        EbN0_dB, BER_sem_diversidade(idx_snr), BER_com_diversidade(idx_snr));
end

%% 8. Geração do Gráfico de Desempenho Analítico
BER_com_diversidade(BER_com_diversidade == 0) = 1e-6; 

if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    try
        graphics_toolkit('gnuplot');
    catch
    end
end

figure('Color', [1 1 1]);
semilogy(vetor_EbN0_dB, BER_sem_diversidade, 'r-o', 'LineWidth', 2, 'MarkerFaceColor', 'r');
hold on;
semilogy(vetor_EbN0_dB, BER_com_diversidade, 'b-s', 'LineWidth', 2, 'MarkerFaceColor', 'b');
grid on;
xlabel('E_b/N_0 (dB)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Taxa de Erro de Bit (BER)', 'FontSize', 11, 'FontWeight', 'bold');
title({'Transceptor OFDM em Canal UWOC Realista'; ...
       'Filtro Passa-Baixas do Emissor + Turbulência por Bolhas'}, ...
       'FontSize', 12, 'FontWeight', 'bold');
legend('1 RX (Com Equalização ZF)', ['Diversidade Espacial MRC (' num2str(N_receptores) ' RXs)'], ...
       'Location', 'SouthWest');