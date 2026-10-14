function [Sinal_ff,sinal_tf,f,df] = FFT_pot2(sinal,ts) 

fs = 1/ts;                  % Taxa de amostragem
ni = length(sinal);         % Tamanho do sinal de entrada
nf = 2^(nextpow2(ni));      % Novo tamanho do sinal

% A transformada via FFT
Sinal_ff = fft(sinal,nf);   %FFT do Sinal de tamanho nf
Sinal_ff = Sinal_ff/fs;     %Fornece o PSD - Power Spectrum Density

% O novo sinal no dominio do tempo
sinal_tf = [sinal,zeros(1,nf-ni)];

% A resolucao na frequencia
df = fs/nf;

% Vetor frequencia
f = (0:df:df*(length(sinal_tf)-1)) - fs/2;