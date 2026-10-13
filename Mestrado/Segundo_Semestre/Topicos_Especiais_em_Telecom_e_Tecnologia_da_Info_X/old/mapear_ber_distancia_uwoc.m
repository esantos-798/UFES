clear; clc; close all;

% Ajuste do toolkit gráfico para Octave
if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    try
        graphics_toolkit('gnuplot');
    catch
    end
end

%% ========================================================================
%% 1. CONFIGURAÇÕES DA VARREDURA DE DISTÂNCIA
%% ========================================================================
vetor_L     = [5, 10, 15, 20, 25]; % Distâncias em metros para avaliar
BER_dist    = zeros(size(vetor_L));

% Parâmetros Fixos do OFDM
N_fft       = 64;       
N_cp        = 16;       
M           = 4;        % QPSK
EbNo_dB     = 16;       % SNR fixa para isolar o efeito da distância
num_blocos  = 300;      

fprintf('Iniciando varredura de desempenho por Distancia (L) a Eb/No = %d dB...\n\n', EbNo_dB);

%% ========================================================================
%% 2. TRANSMISSOR OFDM (DADOS FIXOS PARA COMPARAÇÃO JUSTA)
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
%% 3. LOOP DE DISTÂNCIA (INTERAÇÃO COM O CANAL DE MONTE CARLO)
%% ========================================================================
for idx = 1:length(vetor_L)
    L_atual = vetor_L(idx);
    fprintf('Simulando Canal de Monte Carlo para L = %d metros...\n', L_atual);
    
    % Giarâmetro L modificado dinamicamente sem quebrar o código original
    % Nota: Se preferires, podes alterar a assinatura da tua função original 
    % para "function [h_mc, t_bins] = simular_mc_uwoc_HG_1(L)"
    
    [h_mc_bruto, t_bins] = simular_mc_uwoc_HG_1(L_atual); 
    % Nota técnica: Se o teu ficheiro simular_mc_uwoc_HG_1.m estiver estático com L=20,
    % idealmente deves abrir esse arquivo e mudar a linha "L = 20;" para receber como argumento.
    
    h_canal = h_mc_bruto(:) / sum(h_mc_bruto);

    %% CANAL EM TEMPO REAL
    sinal_filtrado = conv(sinal_tx_serial, h_canal);
    sinal_filtrado = sinal_filtrado(1:length(sinal_tx_serial));
    potencia_sinal = mean(abs(sinal_filtrado).^2);
    
    % Adição de ruído AWGN
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
    BER_dist(idx) = erros / total_bits;
    
    fprintf('Distancia = %2d m | Erros = %5d | BER = %e\n\n', L_atual, erros, BER_dist(idx));
end

%% ========================================================================
%% 4. PLOT DO GRÁFICO (BER vs DISTÂNCIA)
%% ========================================================================
try
    figure(3);
    plot(vetor_L, BER_dist, 'm-s', 'LineWidth', 2, 'MarkerFaceColor', 'm', 'MarkerSize', 8);
    grid on;
    xlabel('Distancia do Canal L (metros)');
    ylabel('Bit Error Rate (BER)');
    title(sprintf('Impacto da Distancia no Canal Otico (OFDM a %d dB)', EbNo_dB));
    
    fprintf('Varredura terminada. Pressione Enter para fechar os graficos...\n');
    pause;
catch
    fprintf('[Aviso]: Interface grafica indisponivel.\n');
end