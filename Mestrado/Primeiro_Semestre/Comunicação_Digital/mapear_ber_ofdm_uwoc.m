clear; clc; close all;

% Ajuste do toolkit gráfico para Octave
if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    try
        graphics_toolkit('gnuplot');
    catch
    end
end

%% ========================================================================
%% 1. CARREGAMENTO DO CANAL ÓTICO SUBAQUÁTICO (UMA ÚNICA VEZ)
%% ========================================================================
fprintf('Executando a simulação de Monte Carlo para o canal UWOC...\n');
[h_mc_bruto, t_bins] = simular_mc_uwoc_HG_1();

% Alinhamento e normalização do canal
h_canal = h_mc_bruto(:); 
h_canal = h_canal / sum(h_canal); 
fprintf('Canal carregado! Iniciando a varredura de Eb/No...\n\n');

%% ========================================================================
%% 2. CONFIGURAÇÕES DA SIMULAÇÃO DE BER
%% ========================================================================
N_fft       = 64;       % Número de subportadoras
N_cp        = 16;       % Prefixo Cíclico
M           = 4;        % Modulação: QPSK (4-QAM)
num_blocos  = 500;      % Mais blocos para conseguir medir BERs baixas (ex: 10^-4)

% Vetor de Eb/No em dB para avaliar
EbNo_dB_vetor = 0:2:28; 
BER_vetor     = zeros(size(EbNo_dB_vetor));

%% ========================================================================
%% 3. TRANSMISSOR OFDM (ESTRUTURA BASE DE DADOS)
%% ========================================================================
bits_por_bloco = N_fft * log2(M);
total_bits     = bits_por_bloco * num_blocos;
bits_tx        = randi([0 1], total_bits, 1);

% Modulação QPSK
bits_agrupados = reshape(bits_tx, log2(M), [])';
simbolos_dec   = bi2de_compativel(bits_agrupados);
constelacao    = [1+1i, -1+1i, -1-1i, 1-1i] / sqrt(2); 
sinal_mapeado  = constelacao(simbolos_dec + 1).';

% Formato de matriz e IFFT
matriz_ofdm    = reshape(sinal_mapeado, N_fft, num_blocos);
sinal_tx_tempo = ifft(matriz_ofdm, N_fft);

% Inserção de Prefixo Cíclico e Serialização
prefixo_ciclico = sinal_tx_tempo(end - N_cp + 1 : end, :);
sinal_tx_cp     = [prefixo_ciclico; sinal_tx_tempo];
sinal_tx_serial = sinal_tx_cp(:);

%% ========================================================================
%% 4. LOOP DE VARREDURA DE Eb/No
%% ========================================================================
% Passa o sinal pelo efeito fixo de filtragem do canal ótico antes do loop
sinal_filtrado_canal = conv(sinal_tx_serial, h_canal);
sinal_filtrado_canal = sinal_filtrado_canal(1:length(sinal_tx_serial));
potencia_sinal       = mean(abs(sinal_filtrado_canal).^2);

for idx = 1:length(EbNo_dB_vetor)
    EbNo_dB = EbNo_dB_vetor(idx);
    
    % Cálculo do Ruído AWGN para este Eb/No específico
    snr = EbNo_dB + 10*log10(log2(M)) + 10*log10(N_fft/(N_fft+N_cp)); 
    potencia_ruido = potencia_sinal / (10^(snr/10));
    
    ruido = sqrt(potencia_ruido/2) * (randn(size(sinal_filtrado_canal)) + 1i*randn(size(sinal_filtrado_canal)));
    sinal_rx_serial = sinal_filtrado_canal + ruido;
    
    %% RECETOR OFDM
    matriz_rx_cp = reshape(sinal_rx_serial, N_fft + N_cp, num_blocos);
    matriz_rx_tempo = matriz_rx_cp(N_cp + 1 : end, :);
    sinal_rx_freq = fft(matriz_rx_tempo, N_fft);
    
    % Igualização ZF com regularização
    H_canal_freq = fft(h_canal, N_fft); 
    H_canal_freq(abs(H_canal_freq) < 1e-4) = 1e-4; % Proteção contra atenuação extrema
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
    
    % Cálculo da BER para este ponto
    erros = sum(bits_tx ~= bits_rx);
    BER_vetor(idx) = erros / total_bits;
    
    fprintf('Eb/No = %2d dB | Erros = %5d | BER = %e\n', EbNo_dB, erros, BER_vetor(idx));
end


%% ========================================================================
%% 5. GERAÇÃO DA CURVA DE DESEMPENHO (BER vs Eb/No)
%% ========================================================================
% Substituição robusta da qfunc(x) usando a função nativa erfc(x/sqrt(2))/2
EbNo_linear = 10.^(EbNo_dB_vetor/10);
x_argumento = sqrt(2 * EbNo_linear);
BER_teorica = 0.5 * erfc(x_argumento / sqrt(2));

try
    figure(2);
    semilogy(EbNo_dB_vetor, BER_teorica, 'r--', 'LineWidth', 2); hold on;
    semilogy(EbNo_dB_vetor, BER_vetor, 'b-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
    grid on;
    axis([min(EbNo_dB_vetor) max(EbNo_dB_vetor) 1e-5 1]);
    
    title('Desempenho de Erro do OFDM no Canal Otico Subaquatico (UWOC)');
    xlabel('Eb/No (dB)');
    ylabel('Bit Error Rate (BER)');
    legend('QPSK Teorico (AWGN Puro)', 'OFDM Igualado ZF (Canal Otico)');
    
    fprintf('\nVarredura concluida. Pressione Enter no terminal para finalizar...\n');
    pause;
catch
    fprintf('\n[Aviso]: Grafico gerado, mas interface indisponivel no terminal.\n');
end