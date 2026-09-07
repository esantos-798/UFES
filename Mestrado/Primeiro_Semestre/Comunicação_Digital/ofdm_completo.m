clc;
clear;
close all;

try
    graphics_toolkit("qt");
catch
    warning("Qt não disponível. Usando toolkit padrão.");
end

%% =====================================================
%% PARÂMETROS
%% =====================================================

N = 64;            % Subportadoras
Ncp = 16;          % Prefixo Cíclico
numSymbols = 500;  % Símbolos OFDM

%% =====================================================
%% GERAÇÃO DOS BITS
%% =====================================================

bits = randi([0 1],2*N*numSymbols,1);

%% =====================================================
%% MODULAÇÃO QPSK
%% =====================================================

bits2 = reshape(bits,2,[]).';

tx_symbols = (2*bits2(:,1)-1) + 1j*(2*bits2(:,2)-1);
tx_symbols = tx_symbols/sqrt(2);

figure;
plot(real(tx_symbols(1:2000)), imag(tx_symbols(1:2000)), '.');
axis([-1.5 1.5 -1.5 1.5]);
grid on;
title('Constelação QPSK Transmitida');
xlabel('In-Phase');
ylabel('Quadrature');

%% =====================================================
%% SERIAL -> PARALELO
%% =====================================================

X = reshape(tx_symbols,N,[]);

%% =====================================================
%% IFFT (MULTIPLEX OFDM)
%% =====================================================

x = ifft(X,N);

%% =====================================================
%% PREFIXO CÍCLICO
%% =====================================================

x_cp = [x(end-Ncp+1:end,:); x];

%% =====================================================
%% SERIALIZAÇÃO
%% =====================================================

tx_signal = x_cp(:);

%% =====================================================
%% SINAL OFDM NO TEMPO
%% =====================================================

figure;
plot(real(tx_signal(1:500)));
grid on;
title('Sinal OFDM no Domínio do Tempo');
xlabel('Amostra');
ylabel('Amplitude');

%% =====================================================
%% ESPECTRO
%% =====================================================

FFTsig = fftshift(abs(fft(tx_signal)));

figure;
plot(FFTsig);
grid on;
title('Espectro OFDM');
xlabel('Frequência');
ylabel('|X(f)|');

%% =====================================================
%% BER x SNR
%% =====================================================

SNRdB = 0:2:30;
BER = zeros(size(SNRdB));

for k=1:length(SNRdB)

    snr = SNRdB(k);

    %% Canal AWGN

    Ps = mean(abs(tx_signal).^2);

    Pn = Ps/(10^(snr/10));

    noise = sqrt(Pn/2)*(randn(size(tx_signal)) ...
          + 1j*randn(size(tx_signal)));

    rx_signal = tx_signal + noise;

    %% Desserialização

    rx_cp = reshape(rx_signal,N+Ncp,[]);

    %% Remove CP

    rx_no_cp = rx_cp(Ncp+1:end,:);

    %% FFT (DEMULTIPLEX)

    Y = fft(rx_no_cp,N);

    %% Paralelo -> Serial

    rx_symbols = Y(:);

    %% Demodulação

    rx_bits = zeros(length(bits),1);

    rx_bits(1:2:end) = real(rx_symbols)>0;
    rx_bits(2:2:end) = imag(rx_symbols)>0;

    %% BER

    BER(k) = sum(bits~=rx_bits)/length(bits);

end

%% =====================================================
%% CONSTELAÇÃO RECEBIDA
%% =====================================================

snr = 10;

Ps = mean(abs(tx_signal).^2);
Pn = Ps/(10^(snr/10));

noise = sqrt(Pn/2)*(randn(size(tx_signal)) ...
      + 1j*randn(size(tx_signal)));

rx_signal = tx_signal + noise;

rx_cp = reshape(rx_signal,N+Ncp,[]);
rx_no_cp = rx_cp(Ncp+1:end,:);

Y = fft(rx_no_cp,N);

rx_symbols = Y(:);

figure;
plot(real(rx_symbols),imag(rx_symbols),'.');
grid on;
title('Constelação Recebida (10 dB)');
xlabel('In-Phase');
ylabel('Quadrature');

%% =====================================================
%% CURVA BER
%% =====================================================

figure;
semilogy(SNRdB,BER,'-o');
grid on;
title('BER x SNR');
xlabel('SNR (dB)');
ylabel('BER');

disp('Simulação concluída.');

disp('Pressione ENTER para fechar...');
pause();