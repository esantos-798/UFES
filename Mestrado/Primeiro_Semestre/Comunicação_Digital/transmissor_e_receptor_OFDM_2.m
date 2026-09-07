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

clc;
clear;
close all;

%% ========================
% Parâmetros do sistema
% ========================
M = 16;                 % Ordem da modulação QAM
B  = 15e6;              % Largura de banda
Rb = 20e6;              % Taxa de bits
SNRdB = 15;             % Faixa de SNR


% ---- Parâmetros Calculados de tempo/frequência do enunciado ----
tau_s = 200e-9;           % Delay spread
Tg    = 4*tau_s;          % Prefixo cíclico = 800 ns
Ts    = 6*Tg;             % Símbolo OFDM total = 4.8 us
Tu    = Ts - Tg;          % Símbolo útil (duração da IFFT) = 4 us
Delta_f = 1/Tu;           % Espaçamento entre subportadoras = 250 kHz
k = log2(M);             % Bits por símbolo (=4)
N  = round(B/Delta_f);    % Número de subportadoras = 60
Nfft  = 2^(nextpow2(N));  % Todas as N subportadoras são usadas (sem nulas)
Ncp = round((Tg*N)/Tu); % Amostras do prefixo cíclico = 12
Fs = 10*B;             % Taxa de amostragem
seq_bit = N*k;         % Bits por símbolo OFDM
Delta_t = 1/Fs;           % Amostragem de tempo

fprintf('--- Parâmetros calculados ---\n');
fprintf('Tg   = %.0f ns\n', Tg*1e9);
fprintf('Ts   = %.2f us\n', Ts*1e6);
fprintf('Tu   = %.2f us\n', Tu*1e6);
fprintf('Delta_f = %.0f kHz\n', Delta_f/1e3);
fprintf('N (Nfft) = %d\n', N);
fprintf('Ncp  = %d amostras\n', Ncp);
fprintf('Fs   = %.2f MHz\n', Fs/1e6);
fprintf('Bits por símbolo OFDM = %d\n\n', seq_bit);
fprintf('Delta t = %d\n\n', Delta_t);


% Geração dos bits de entrada do transmissor
x = randi([0, M-1], 1, N);

%======TX=======

% Conversão Série-Paralelo
x = x.';

% Mapeamento dos bits em simbolos
if exist('qammod', 'file') == 2 || exist('qammod', 'builtin')
    disp('-> Cenário: Pacote instalado! Usando qammod nativa.');
    Y = qammod(x, M);
else
    disp('-> Cenário: Sem pacote! Usando a função custom como Fallback.');
    % Chama a função que você mesmo escreveu
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

%======Canal AWGN=======
if exist('awgn', 'file') == 2 || exist('awgn', 'builtin')
    disp('-> Cenário: Pacote instalado! Usando awgn nativa.');
    y_r = awgn(y_ig, SNRdB, "measured");
else    
    disp('-> Cenário: Sem pacote! Usando a função custom como Fallback.');
    % Chama a função que você mesmo escreveu
    y_r = awgn_custom(y_ig, SNRdB, "measured");
end
%======RX=======

% Conversão Série-Paralelo
y_r = y_r.';

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
    disp('-> Cenário: Pacote instalado! Usando qamdemod nativa.');
    xr = qamdemod(y_rx, M);
else
    disp('-> Cenário: Sem pacote! Usando a função custom como Fallback.');
    % Chama a função que você mesmo escreveu
    xr = custom_qamdemod(y_rx, M);
end

% Analise de Desempenho
if exist('biterr', 'file') == 2 || exist('biterr', 'builtin')
    disp('-> Cenário: Pacote instalado! Usando biterr nativa.');
    [number_errors, BER_ratio] = biterr(x,xr);
else
    disp('-> Cenário: Sem pacote! Usando a função custom como Fallback.');
    % Chama a função que você mesmo escreveu
    [number_errors, BER_ratio] = custom_biterr(x,xr);
    posicao_erro = find(x ~= xr);
    disp(['O erro ocorreu no bit número: ', posicao_erro]);
end

t_util = 0 : Delta_t : (N-1)*Delta_t;
t_simbolo = 0 : Delta_t : ((N+Ncp)-1)*Delta_t;

resolucao_frequencia = Fs / Nfft;
vetor_frequencia = (-Nfft/2 : Nfft/2 - 1) * resolucao_frequencia;

% Potência linear normalizada pelo tamanho da FFT
Potencia_Linear = (abs(Y_rx).^2) / (Nfft^2)

% Densidade de potência (dividido pela resolução em Hz)
PSD_linear = Potencia_Linear / resolucao_frequencia

% Conversão para dB/Hz
if exist('db', 'file') == 2 || exist('db', 'builtin')
    disp('-> Cenário: Pacote instalado! Usando db nativa.');
    PSD_dB_Hz = db(PSD_linear);
else
    disp('-> Cenário: Sem pacote! Usando a função custom como Fallback.');
    PSD_dB_Hz = custom_db(PSD_linear)
end

% --- CONFIGURAÇÃO DA VARREDURA DE OSNR ---
OSNR_dB_vetor = 0 : 2 : SNRdB+1;              % Valores de OSNR em dB para testar
BER_final = zeros(size(OSNR_dB_vetor));  % Vetor para guardar os resultados

for idx = 1:length(OSNR_dB_vetor)
    OSNR_atual = OSNR_dB_vetor(idx)
    
    % [1] --- TRANSMISSOR ---
    % bits -> qammod -> mapeamento (64) -> ifft -> prefixo ciclico (sinal_tx)
    
    % [2] --- CANAL COM RUÍDO ---
    % Aqui adicionamos ruído baseado no OSNR desejado.
    % Se não tiver a função 'awgn', podemos gerar o ruído gaussiano na unha:
    % P_sinal = mean(abs(sinal_tx).^2);
    % P_ruido = P_sinal / (10^(OSNR_atual/10));
    % ruido = sqrt(P_ruido/2) * (randn(size(sinal_tx)) + 1i*randn(size(sinal_tx)));
    % sinal_com_ruido = sinal_tx + ruido;
    
    % [3] --- RECEPTOR ---
    % retira prefixo -> fft -> demodula -> bits_recebidos
    
    % [4] --- CÁLCULO DA BER ---
    if exist('biterr', 'file') == 2 || exist('biterr', 'builtin')
        [~, BER_ratio] = biterr(x, xr);
    else
        [~, BER_ratio] = custom_biterr(x, xr)
        posicao_erro = find(x ~= xr);
    end        
    BER_final(idx) = BER_ratio;
end


%======Plotagem de sinais e espectros=======
if exist('scatterplot', 'file') == 2 || exist('scatterplot', 'builtin')
    disp('-> Cenário: Pacote instalado! Usando scatterplot nativa.');
    figure
    scatterplot(Y)
    scatterplot(y_rx)
else
    disp('-> Cenário: Sem pacote! Usando a função custom como Fallback.');
    custom_scatterplot(Y)
    custom_scatterplot(y_rx)
end

figure
subplot(2,1,1)
plot(real(y_ig))
subplot(2,1,2);
plot(imag(y_ig))


figure
if exist('db', 'file') == 2 || exist('db', 'builtin')
    disp('-> Cenário: Pacote instalado! Usando db nativa.');
    plot(db(abs(fftshift(fft(y_ig)))))
else
    disp('-> Cenário: Sem pacote! Usando a função custom como Fallback.');
    plot(custom_db(abs(fftshift(fft(y_ig)))))
end    


figure;
plot(vetor_frequencia / 1e6, PSD_dB_Hz, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Frequência (MHz)');
ylabel('Densidade Espectral de Potência (dB/Hz)');
title('Espectro do Sinal OFDM Transmitido');
%$pause();

figure;
semilogy(OSNR_dB_vetor, BER_final, 'b-o', 'LineWidth', 2);
grid on;
xlabel('OSNR (dB)');
ylabel('Taxa de Erro de Bit (BER)');
title('Curva de BER vs OSNR do Sistema OFDM');
ylim([1e-6 1]); % Ajusta o limite inferior para ver a "cascata" cair

drawnow; 

input('Pressione ENTER para fechar o grafico e encerrar...', 's');