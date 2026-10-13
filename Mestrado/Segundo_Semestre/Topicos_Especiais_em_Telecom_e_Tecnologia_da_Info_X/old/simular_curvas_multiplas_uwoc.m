clear; clc; close all;

% Ajuste do toolkit gráfico para Octave
if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    try
        graphics_toolkit('gnuplot');
    catch
    end
end

%% ========================================================================
%% 1. CONFIGURAÇÕES DA SIMULAÇÃO
%% ========================================================================
N_fft       = 64;       % Número de subportadoras
N_cp        = 16;       % Tamanho do Prefixo Cíclico
M           = 4;        % QPSK
num_blocos  = 400;      % Blocos por simulação

EbNo_dB_vetor = 0:3:27; % Varredura de SNR
distancias    = [5, 15, 25]; % As três distâncias físicas para comparar

% Cores e marcadores para o gráfico
estilos = {'b-o', 'g-s', 'm-^'}; 

%% ========================================================================
%% 2. TRANSMISSOR OFDM (DADOS FIXOS)
%% ========================================================================
bits_por_bloco = N_fft * log2(M);
total_bits     = bits_por_bloco * num_blocos;
bits_tx        = randi([0 1], total_bits, 1);

% Modulação QPSK
bits_agrupados = reshape(bits_tx, log2(M), [])';
simbolos_dec   = bi2de_compativel(bits_agrupados);
constelacao    = [1+1i, -1+1i, -1-1i, 1-1i] / sqrt(2); 
sinal_mapeado  = constelacao(simbolos_dec + 1).';

% Matriz OFDM e IFFT
matriz_ofdm    = reshape(sinal_mapeado, N_fft, num_blocos);
sinal_tx_tempo = ifft(matriz_ofdm, N_fft);
prefixo_ciclico = sinal_tx_tempo(end - N_cp + 1 : end, :);
sinal_tx_cp     = [prefixo_ciclico; sinal_tx_tempo];
sinal_tx_serial = sinal_tx_cp(:);

%% ========================================================================
%% 3. EXECUÇÃO DO LOOP DA MATRIZ DE CENÁRIOS
%% ========================================================================
figure(4);
% Primeiro, plota a curva teórica AWGN para referência
EbNo_linear = 10.^(EbNo_dB_vetor/10);
BER_teorica = 0.5 * erfc(sqrt(2 * EbNo_linear) / sqrt(2));
semilogy(EbNo_dB_vetor, BER_teorica, 'r--', 'LineWidth', 2); hold on;

for d_idx = 1:length(distancias)
    L_atual = distancias(d_idx);
    fprintf('\n--- Simulando Canal Físico para L = %d metros ---\n', L_atual);
    
    % Chama o Monte Carlo para a distância do loop
    [h_mc_bruto, t_bins] = simular_mc_uwoc_HG_1(L_atual);
    h_canal = h_mc_bruto(:) / sum(h_mc_bruto);
    
    % Filtra o sinal pelo canal óptico
    sinal_filtrado = conv(sinal_tx_serial, h_canal);
    sinal_filtrado = sinal_filtrado(1:length(sinal_tx_serial));
    potencia_sinal = mean(abs(sinal_filtrado).^2);
    
    BER_resultado = zeros(size(EbNo_dB_vetor));
    
    for snr_idx = 1:length(EbNo_dB_vetor)
        EbNo_dB = EbNo_dB_vetor(snr_idx);
        
        % Adição de ruído AWGN baseado no Eb/No
        snr = EbNo_dB + 10*log10(log2(M)) + 10*log10(N_fft/(N_fft+N_cp)); 
        potencia_ruido = potencia_sinal / (10^(snr/10));
        ruido = sqrt(potencia_ruido/2) * (randn(size(sinal_filtrado)) + 1i*randn(size(sinal_filtrado)));
        sinal_rx_serial = sinal_filtrado + ruido;
        
        %% RECEPTOR OFDM
        matriz_rx_cp = reshape(sinal_rx_serial, N_fft + N_cp, num_blocos);
        matriz_rx_tempo = matriz_rx_cp(N_cp + 1 : end, :);
        sinal_rx_freq = fft(matriz_rx_tempo, N_fft);
        
        % Igualização ZF
        H_canal_freq = fft(h_canal, N_fft);
        H_canal_freq(abs(H_canal_freq) < 1e-4) = 1e-4;
        H_matriz = repat_compativel(H_canal_freq, num_blocos);
        
        sinal_rx_igualizado = sinal_rx_freq ./ H_matriz;
        sinal_rx_final = sinal_rx_igualizado(:);
        
        %% DEMAPEAMENTO
        bits_rx = zeros(total_bits, 1);
        idx_bit = 1;
        for i = 1:length(sinal_rx_final)
            if real(sinal_rx_final(i)) >= 0 && imag(sinal_rx_final(i)) >= 0
                dec = [0; 0]; 
            elseif real(sinal_rx_final(i)) < 0 && imag(sinal_rx_final(i)) >= 0
                dec = [0; 1]; 
            elseif real(sinal_rx_final(i)) < 0 && imag(sinal_rx_final(i)) < 0
                dec = [1; 0]; 
            else
                dec = [1; 1]; 
            end
            bits_rx(idx_bit:idx_bit+1) = dec;
            idx_bit = idx_bit + 2;
        end
        
        erros = sum(bits_tx ~= bits_rx);
        BER_resultado(snr_idx) = erros / total_bits;
    end
    
    % Plota a linha desta distância específica
    semilogy(EbNo_dB_vetor, BER_resultado, estilos{d_idx}, 'LineWidth', 2, 'MarkerFaceColor', estilos{d_idx}(1));
end

%% ========================================================================
%% 4. FORMATAÇÃO DO GRÁFICO FINAL
%% ========================================================================
try
    grid on;
    axis([0 27 1e-4 1]);
    xlabel('OSNR (dB) [Referência Equivalente em Banda Base]');
    ylabel('Bit Error Rate (BER)');
    title('Curvas de BER vs OSNR no Canal Ótico Subaquático (OFDM-UWOC)');
    
    legend('AWGN Teorico (Ideal)', 'L = 5 metros', 'L = 15 metros', 'L = 25 metros', 'Location', 'southwest');
    
    fprintf('\nSimulacao concluida com sucesso. Pressione Enter para fechar...\n');
    pause;
catch
    fprintf('[Aviso]: Interface grafica indisponivel.\n');
end