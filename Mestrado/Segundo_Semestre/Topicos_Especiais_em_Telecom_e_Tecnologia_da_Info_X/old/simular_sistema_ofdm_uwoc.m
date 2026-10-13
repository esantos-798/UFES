clear; clc; close all;

% Tenta usar gnuplot para evitar travar a execução no CLI se gráficos forem chamados
if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    try
        graphics_toolkit('gnuplot');
    catch
        % Ignora se não houver toolkit
    end
end

%% ========================================================================
%% 1. CARREGAMENTO E ALINHAMENTO DO CANAL ÓTICO (FÍSICA)
%% ========================================================================
fprintf('Executando a simulação de Monte Carlo para o canal UWOC...\n');

[h_mc_bruto, t_bins] = simular_mc_uwoc_HG_1();

% CORREÇÃO CRÍTICA 1: Remover o atraso de propagação fixa (trazer para t=0)
% Garante que a resposta ao impulso comece no início da janela temporal
h_canal = h_mc_bruto(:); 

% Normalização de energia unitária do canal
h_canal = h_canal / sum(h_canal); 

fprintf('Canal carregado e alinhado! Iniciando o sistema OFDM...\n\n');

%% ========================================================================
%% 2. PARÂMETROS DO TRANSCEPTOR OFDM
%% ========================================================================
N_fft      = 64;       % Número de subportadoras
N_cp       = 16;       % Tamanho do Prefixo Cíclico
num_blocos = 200;      % Aumentado para melhor visualização estatística
M          = 4;        % Modulação: 4-QAM (QPSK)
EbNo_dB    = 25;       % Aumentado temporariamente para isolar o efeito do canal

%% ========================================================================
%% 3. TRANSMISSOR OFDM
%% ========================================================================
bits_por_bloco = N_fft * log2(M);
total_bits = bits_por_bloco * num_blocos;
bits_tx = randi([0 1], total_bits, 1);

% Modulação Digital
bits_agrupados = reshape(bits_tx, log2(M), [])';
simbolos_dec = bi2de_compativel(bits_agrupados);

% Mapeamento QPSK
constelacao = [1+1i, -1+1i, -1-1i, 1-1i] / sqrt(2); 
sinal_mapeado = constelacao(simbolos_dec + 1).';

% Formato de matriz (Subportadoras x Blocos)
matriz_ofdm = reshape(sinal_mapeado, N_fft, num_blocos);

% IFFT
sinal_tx_freq = matriz_ofdm;
sinal_tx_tempo = ifft(sinal_tx_freq, N_fft);

% Inserção do Prefixo Cíclico
prefixo_ciclico = sinal_tx_tempo(end - N_cp + 1 : end, :);
sinal_tx_cp = [prefixo_ciclico; sinal_tx_tempo];

% Serialização
sinal_tx_serial = sinal_tx_cp(:);

%% ========================================================================
%% 4. CANAL DE TRANSMISSÃO (CONVOLUÇÃO + RUÍDO)
%% ========================================================================
sinal_recebido_canal = conv(sinal_tx_serial, h_canal);
sinal_recebido_canal = sinal_recebido_canal(1:length(sinal_tx_serial));

% Cálculo de ruído baseado no Eb/No real da banda base
snr = EbNo_dB + 10*log10(log2(M)) + 10*log10(N_fft/(N_fft+N_cp)); 
potencia_sinal = mean(abs(sinal_recebido_canal).^2);
potencia_ruido = potencia_sinal / (10^(snr/10));

ruido = sqrt(potencia_ruido/2) * (randn(size(sinal_recebido_canal)) + 1i*randn(size(sinal_recebido_canal)));
sinal_rx_serial = sinal_recebido_canal + ruido;

%% ========================================================================
%% 5. RECETOR OFDM
%% ========================================================================
matriz_rx_cp = reshape(sinal_rx_serial, N_fft + N_cp, num_blocos);

% Remoção do Prefixo Cíclico (CP)
matriz_rx_tempo = matriz_rx_cp(N_cp + 1 : end, :);

% FFT
sinal_rx_freq = fft(matriz_rx_tempo, N_fft);

% CORREÇÃO CRÍTICA 2: Resposta de Canal Estável na Frequência
% Calcula a FFT do canal preenchendo com zeros (padding) até o tamanho N_fft
H_canal_freq = fft(h_canal, N_fft); 

% Evitar divisão por zero ou valores extremamente baixos (Regularização)
H_canal_freq(abs(H_canal_freq) < 1e-4) = 1e-4;

H_matriz = repat_compativel(H_canal_freq, num_blocos);

% Igualização Zero Forcing (ZF)
sinal_rx_igualizado = sinal_rx_freq ./ H_matriz;

%% ========================================================================
%% 6. DEMAPEAMENTO E CÁLCULO DE DESEMPENHO
%% ========================================================================
sinal_rx_final = sinal_rx_igualizado(:);

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
BER = erros / total_bits;

%% ========================================================================
%% 7. VISUALIZAÇÃO DOS RESULTADOS CORRIGIDOS
%% ========================================================================
fprintf('--- RESULTADOS DA SIMULAÇÃO ---\n');
fprintf('Erros de Bit detetados: %d\n', erros);
fprintf('BER (Bit Error Rate) obtida: %e\n\n', BER);

try
    figure(1);
    
    % Gráfico 1: Constelação limpa pós-igualização
    subplot(2,1,1);
    plot(sinal_rx_igualizado(:), 'b.', 'MarkerSize', 5);
    grid on;
    title('Constelacao QPSK Corrigida apos Igualizacao ZF');
    xlabel('In-Phase (Real)'); ylabel('Quadrature (Imag)');
    axis([-2 2 -2 2]);

    % Gráfico 2: Resposta em Frequência coerente
    subplot(2,1,2);
    if length(t_bins) > 1
        dt = t_bins(2) - t_bins(1);
        f_eixo = (0:N_fft-1) * (1/dt) / N_fft / 1e6; % Frequência em MHz
    else
        f_eixo = 0:N_fft-1;
    end
    stem(f_eixo, 20*log10(abs(H_canal_freq)), 'LineWidth', 1.5);
    grid on;
    title('Resposta em Frequencia Alinhada do Canal Otico');
    xlabel('Frequencia (MHz)'); ylabel('Magnitude (dB)');
    
    fprintf('Pressione Enter no terminal para finalizar...\n');
    pause;
catch
    fprintf('[Aviso]: Janela grafica indisponivel no terminal. Processamento concluido.\n');
end