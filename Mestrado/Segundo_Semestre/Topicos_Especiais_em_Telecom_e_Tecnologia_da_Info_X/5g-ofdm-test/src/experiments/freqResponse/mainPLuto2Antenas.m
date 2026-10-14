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

% Signal Parameters
L = 1024;                   % Buffer Length
Fs = 10e6;                  % Baseband sample rate
T = 1/Fs;                   % Sampling Time
t = linspace(0,T*L,L);      % Time vector
fc = 2397e6;                 % Center Frequency
ftone = 100e3;

% Pluto Parameters
pGain = -30;
pRadioIDTx = "usb:0";                % Transmitter Radio IP
pRadioIDRx = "usb:1";                % Receiver Radio IP

% Create a tx pluto object
load plutoTxFilter
txPluto = sdrtx('Pluto','RadioID',pRadioIDTx,...
          'CenterFrequency',fc, ...
          'BasebandSampleRate',Fs,'Gain',pGain,...
          'ShowAdvancedProperties',true,...
          'UseCustomFilter',true,filtTxConfig{:});
txPluto.Gain = 0;
% Create a rx pluto object
load plutoRxFilter
rxPluto = sdrrx('Pluto',...
       'RadioID',pRadioIDRx,...
       'CenterFrequency',fc,...
       'BasebandSampleRate',Fs,...
       'SamplesPerFrame', 20*L,...
       'OutputDataType','double',...
       'UseCustomFilter',true,filtRxConfig{:}); 
rxPluto.GainSource = 'Manual';
rxPluto.Gain = 60;

% Generate Step Ones signal
stepF = exp(j*2*pi*ftone*t);
stepF = stepF.';

% Simulation Parameters
fmax = 2403e6;                              % Max Frequency Search
fmin = 2397e6;                              % Start Frequency
fpass = 0.1e6;                               % Frequency change
f=fmin;                                     % Initiate f vector
out = [];                                   % Output buffer

while f<fmax-1

    % Display current value
    fprintf('Current frequency = %d \n',f);
    
    % Transmit Pluto ones signal
    txPluto.transmitRepeat(stepF);
    
    % Receive Pluto signal
    [y,datavalid,overflow] = rxPluto();
    
    % Calculate spectrum
    Y = fft(y);
    P2 = abs(Y/(L/2));
    P1 = P2(1:L/2);
    fvec = linspace(0,Fs/2,L/2);
    
    % Partial plot results for each iteration
    % Time Domain Plot
    subplot(2,1,1);
    plot(real(y));
    xlabel('Time (s)')
    ylabel('u.A');
    title('Time Domain Signal Tone')
    
    % Frequency Domain Plot
    subplot(2,1,2);
    plot(fvec,P1);
    xlabel('Frequency (Hz)')
    ylabel('Maginitude u.A');
    title('Frequency Domain Magnitude Spectrum')
    xlim([0 Fs/2])
    
    % Validate measurements - When DAC slips the spectrum is NaN number
    if max(abs(y)) < 1.5 && ~isnan(max(abs(P1)))
        out = [out max(abs(y))];
        pause(0.1);
        f = f + fpass;
        
        % Stop Pluto Transmission and reception
        release(txPluto);
        release(rxPluto);
        
        % Set new Center Frequency
        txPluto.CenterFrequency = f;
        rxPluto.CenterFrequency = f;
    end
end

% Normalize received magnitudes
out = out/max(abs(out));

% Convert to Logarihtimic scale
outLog = 10*log(out);

load ('OFDM channel.mat');
plotChannel(upHest,2.4e9,15e3);
hold on
% Plot results
%figure();
fvec = linspace(fmin,fmax,length(out));
plot(fvec/1e6,outLog);
%ylim([-20,0]);
xlabel('Frequency MHz');
ylabel('Normalized u.A (dB)');
title('ADALM Pluto Frequency Response');
grid;