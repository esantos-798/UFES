clc;
clearvars;
close all;

% Add function paths
addpath('../../adalm2000');
addpath('../../adalm-pluto');
addpath('../../etc');
addpath('../../filters');
addpath('../../newradio');

% Signal Parameters
% Buffer Length

Fs = 61.44e6;                 % Baseband sample rate
ftone = 1e6;

% Transmission parameters
L = 100*round(Fs/ftone);                        % Buffer length to new tone
T = 1/Fs;                                   % Sampling Time
t = (0:L-1)*T;                              % Time vector to new tone

% Generate new Tone
%S1 = sin(2*pi*ftone*t);
%S1 = S1';
%stepF = complex(S1);
S1 = exp(j*2*pi*ftone*t');


y = [S1;S1];

 % Calculate spectrum
Y = fft(y);
P2 = abs(Y/(L));
%P2 = fftshift(P2);
P1 = P2(1:L);
fvec = linspace(0,Fs/2,L);


 % Partial plot results for each iteration
    % Time Domain Plot
    subplot(2,1,1);
    plot(real(y));
    xlabel('Time (s)')
    ylabel('u.A');
    title('Time Domain Signal Tone')
     ylim([-2,2])
    % Frequency Domain Plot
    subplot(2,1,2);
    plot(fvec,P1);
    xlabel('Frequency (Hz)')
    ylabel('Maginitude u.A');
    title('Frequency Domain Magnitude Spectrum')
    xlim([0 Fs/2]);
    ylim([0,1]);