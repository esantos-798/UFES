% Test ADALM2000 frequency response
% Developed by Augusto Peterle
% 17/05/2023

clc;
clearvars;
close all;

% Add function paths
addpath('../../adalm2000');
addpath('../../adalm-pluto');
addpath('../../etc');
addpath('../../filters');
addpath('../../newradio');

L = 1024;                   % Buffer Length
Fs = 512*15e3;              % Baseband sample rate
T = 1/Fs;                   % Sampling Time
t = linspace(0,T*L,L);      % Time vector
fc = 2.4e9;

% Generate Impulse at zero
stepF = t>=0;
stepF = double(stepF);

% Passband Sample Rate
Fspb = 2.4e11;
[y,tt] = modulate(stepF,fc,Fspb);

 % Calculate spectrum
Y = fft(y);
P2 = abs(Y/(L/2));
P1 = P2(1:L/2);
fvec = linspace(0,Fspb/2,L/2);

% Partial plot results for each iteration
% Time Domain Plot
subplot(2,1,1);
plot(t,y);
xlabel('Time (s)')
ylabel('u.A');
title('Time Domain Signal Tone')

% Frequency Domain Plot
subplot(2,1,2);
plot(fvec,P1);
xlabel('Frequency (Hz)')
ylabel('Maginitude u.A');
title('Frequency Domain Magnitude Spectrum')
xlim([0 Fspb/2])