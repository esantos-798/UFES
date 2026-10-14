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
stepF = S1;

% Pluto Parameters
pGain = 0;
rxGain = 60;
pRadioIDTx = 'usb:0';           % Transmitter Radio ID
pRadioIDRx = 'usb:1';           % Receiver Radio ID

% Simulation Parameters
fmax = 5.95e9;                            % Max Frequency Search
%fmax = 3.8e9;
fmin = 325e6;                             % Start Frequency
fpass = 20e6;                             % Frequency change


% Create a tx pluto object
load plutoTxFilter
load filtroTxFreqTest
filtroTxFreqTest = filtnv;
txPluto = sdrtx('Pluto','RadioID',pRadioIDTx,...
    'CenterFrequency',fmin, ...
    'BasebandSampleRate',Fs,'Gain',pGain,...
    'ShowAdvancedProperties',true,...
    filtroTxFreqTest{:});

% Create a rx pluto object
load plutoRxFilter
load filtroRxFreqTest
filtroRxFreqTest = filtnv;
rxPluto = sdrrx('Pluto',...
    'RadioID',pRadioIDRx,...
    'CenterFrequency',fmin,...
    'BasebandSampleRate',Fs,...
    'SamplesPerFrame', 2*L,...
    'OutputDataType','double', ...
    filtroRxFreqTest{:});

%rxPluto.GainSource = 'Manual';
rxPluto.GainSource = 'Manual';
rxPluto.Gain = rxGain;

%configurePlutoRadio('AD9364','usb:0')
%configurePlutoRadio('AD9364','usb:1')

f=fmin;                                     % Initiate f vector
out = [];                                   % Output buffer
txPluto.transmitRepeat(stepF);

while f<fmax-1

    % Display current value
    fprintf('Current frequency = %d MHz \n',f/1e6);

    % Transmit Pluto ones signal
    txPluto.transmitRepeat(stepF);
    pause(0.1);

    % Receive Pluto signal
    [y,datavalid,overflow] = rxPluto();

    % Calculate spectrum
    Y = fft(y);
    P2 = abs(Y/(L));
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


    % Validate measurements - When DAC slips the spectrum is NaN number
    if max(abs(y)) < 1.5 && ~isnan(max(abs(P1)))
        out = [out max(abs(P1))];
        %out = [out max(abs(y))];
        pause(0.1);
        f = f + fpass;

        % Stop Pluto Transmission and reception
        release(txPluto);
        release(rxPluto);

        %     Create a tx pluto object
        txPluto = sdrtx('Pluto','RadioID',pRadioIDTx,...
            'CenterFrequency',f, ...
            'BasebandSampleRate',Fs,'Gain',pGain,...
            'ShowAdvancedProperties',true,...
            filtroTxFreqTest{:});

        %     Create a rx pluto object
        rxPluto = sdrrx('Pluto',...
            'RadioID',pRadioIDRx,...
            'CenterFrequency',f,...
            'BasebandSampleRate',Fs,...
            'SamplesPerFrame', 2*L,...
            'OutputDataType','double',...
            filtroRxFreqTest{:});
        rxPluto.GainSource = 'Manual';
        rxPluto.Gain = rxGain;




    end
end

% Normalize received magnitudes
out = out/max(abs(out));

% Convert to Logarihtimic scale
outLog = 10*log(out);
outLog = smooth(outLog);
outLog = outLog - max(outLog);

% Plot results
figure();
fvec = linspace(fmin,fmax,length(out));
plot(fvec(2:end)/1e6,outLog(2:end));
ylim([-80,0]);
xlabel('Frequency MHz');
ylabel('Normalized u.A (dB)');
title('ADALM Pluto Frequency Response');
grid;