%% ==========================================================
%  SISTEMA OFDM 16-QAM EM CANAL AWGN
%  Geração da curva BER x SNR
%
%  Parâmetros de projeto (dados de entrada):
%    Rb   = 20 Mbps
%    B    = 15 MHz
%    tau_s (delay spread) = 200 ns
%    16-QAM
%    Tg (CP)          = 4 x tau_s = 800 ns
%    Ts (símbolo total)= 6 x Tg   = 4.8 us
%    Tu (símbolo útil) = Ts - Tg  = 4 us
%    Delta_f = 1/Tu = 250 kHz
%    N = B/Delta_f = 60 subportadoras (todas usadas, sem bandas nulas)
%    seq = k x N = 4 x 60 = 240 bits por símbolo OFDM
% ===========================================================
graphics_toolkit('gnuplot'); % Usa o motor gráfico leve e universal

% Verifica se estamos rodando no Octave antes de mexer no toolkit gráfico
if exist('OCTAVE_VERSION', 'builtin')
    % Se for Octave, silencia os warnings gráficos chatos
    warning('off', 'all'); 
end

close all;
clc;
clear all;


% ========================
% Parâmetros do sistema
% ========================
M = 16;                 % Ordem da modulação QAM
B  = 15e6;              % Largura de banda
Rb = 20e6;              % Taxa de bits
SNRdB = 15;             % Faixa de SNR
SNRMax = 15;              % SNRMax para gráfico


% ---- Parâmetros Calculados de tempo/frequência do enunciado ----
tau_s = 500e-9;           % Delay spread
Tg    = 4*tau_s;          % Prefixo cíclico = 800 ns
Ts    = 6*Tg;             % Símbolo OFDM total = 4.8 us
Tu    = Ts - Tg;          % Símbolo útil (duração da IFFT) = 4 us
Delta_f = 1/Tu;           % Espaçamento entre subportadoras = 250 kHz
k = log2(M);              % Bits por símbolo (=4)
N  = round(B/Delta_f);    % Número de subportadoras = 60
Nfft  = 2^(nextpow2(N));  % Todas as N subportadoras são usadas (sem nulas)
Ncp = round((Tg*N)/Tu);   % Amostras do prefixo cíclico = 12
Fs = 10*B;                % Taxa de amostragem
seq_bit = N*k;            % Bits por símbolo OFDM
tam_PDU = 1500;           % Qtd de sinais OFDM a serem transmitidos

fprintf('--- Parâmetros calculados ---\n');
fprintf('Tg   = %.0f ns\n', Tg*1e9);
fprintf('Ts   = %.2f us\n', Ts*1e6);
fprintf('Tu   = %.2f us\n', Tu*1e6);
fprintf('Delta_f = %.0f kHz\n', Delta_f/1e3);
fprintf('N (Nfft) = %d\n', N);
fprintf('Ncp  = %d amostras\n', Ncp);
fprintf('Fs   = %.2f samples/s\n', Fs/1e6);
fprintf('Bits por símbolo OFDM = %d\n\n', seq_bit);

y_ig_PDU = [];
x_total = [];
for ii = 1:tam_PDU
    % Geração dos bits de entrada do transmissor
    x = randi([0, M-1], 1, N);
    x_total = [x_total, x];

    %======TX=======

    % Conversão Série-Paralelo
    x = x.';

    % Mapeamento dos bits em simbolos
    if exist('qammod', 'file') == 2 || exist('qammod', 'builtin')
        Y = qammod(x, M);
    else
        Y = custom_qammod(x, M);
    end

    % Interpolação (escalonamento)
    Nfft2 = 2*Nfft; % para melhor visualização
    y_tx = zeros(Nfft,1);
    y_tx(1:N/2) = Y(1:N/2);
    y_tx(Nfft-((N/2)-1):Nfft) = Y((N/2)+1:end);

    % Multiplexação via IDFT
    y = ifft(y_tx);

    % Insere Intervalo de Guarda
    y_ig = [y(end+1-Ncp:end); y];

    % Conversão paralelo-Série
    y_ig = y_ig.';

    % Concatena os sinais OFDM
    y_ig_PDU = [y_ig_PDU, y_ig];
end


%====== Cálculo das BERs x SNR ====================
% Loop para cada valor de SNR
% Vetor de SNR (1 até SNRMax)
SNR = 1:SNRMax;
for jj = 1:length(SNR)

    %======Canal AWGN=======

    
    SNR_dB_ofdm = SNR(jj) - 10*log10(Nfft/N) + 10*log10(k);

    if exist('awgn', 'file') == 2 || exist('awgn', 'builtin')
        y_r = awgn(y_ig_PDU, SNR_dB_ofdm, "measured");
    else    
        y_r = awgn_custom(y_ig_PDU, SNR_dB_ofdm, "measured");
    end
    
    %======RX=======
    y_r_mat = reshape(y_r,[length(y_ig), tam_PDU]);
    xr_total = [];
    for kk = 1:tam_PDU
        % Conversão Série-Paralelo
        y_r = y_r_mat(:,kk); % já paralelizado
        %y_r = y_r_kk.';

        % Remoção do Intervalo de Guarda
        y_r_sem_ig = y_r(Ncp+1:end);

        % Demultiplexação via DFT
        Y_rx = fft(y_r_sem_ig);

        % Deinterpolação (descalonamento)
        y_rx = zeros(N, 1);
        y_rx(1:N/2) = Y_rx(1:N/2);
        y_rx((N/2)+1:end) = Y_rx((Nfft/2)+((Nfft - N)/2)+1:end);

        % Demapamento dos Simbolos em Bits
        if exist('qamdemod', 'file') == 2 || exist('qamdemod', 'builtin')
            xr = qamdemod(y_rx, M);
        else
        xr = custom_qamdemod(y_rx, M);
        end

         % Conversão Série-Paralelo
         xr = xr.';

        xr_total = [xr_total, xr];
    end
    
    % Cálculo da BER
    if exist('biterr', 'file') == 2 || exist('biterr', 'builtin')
        [~, BER(jj)] = biterr(x_total, xr_total);
    else
        [~, BER(jj)] = custom_biterr(x_total, xr_total);
    end
end

% ====== Plot da curva BER x SNR ===============
figure;
semilogy(SNR, BER, 'o', 'LineWidth', 1.5);
xlabel('SNR (dB)');
ylabel('BER');
title('Curva BER x SNR');
grid on
hold on


%BERxSNR Teorica

% snr_vetor = 1:SNRMax;
% snr_linear = 10.^(snr_vetor / 10);
% BER_teorica = 0.5 * erfc(sqrt(snr_linear)); 
% semilogy(snr_vetor, BER_teorica, 'r--', 'LineWidth', 2); % Teoria
% --- CÁLCULO DA BER TEÓRICA ---
if exist('berawgn', 'file') == 2 || exist('berawgn', 'builtin')
    ber_teorica = berawgn(SNR, 'qam', M);
else
    ber_teorica = ber_teorica_custom(SNR, M);
end
semilogy(SNR, ber_teorica, 'g-', 'LineWidth', 2); % Teoria



