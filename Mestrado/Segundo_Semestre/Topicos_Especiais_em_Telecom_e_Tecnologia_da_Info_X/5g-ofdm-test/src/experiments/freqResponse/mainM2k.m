% Test ADALM2000 frequency response
% Developed by Augusto Peterle
% 16/05/2023
import clib.libm2k.libm2k.*
clc;
clearvars;
close all;

% Add function paths
addpath('../../adalm2000');
addpath('../../adalm-pluto');
addpath('../../etc');
addpath('../../filters');
addpath('../../newradio');

%% ADALM 2000 Parameters
pM2K.pDACSampleRate = 75e6;              % DAC Available Sample Rate
pM2K.pADCSampleRate = 100e6;             % ADC Available Sample Rate

% Open a TX instance
m2k = context.m2kOpen();

% If NULL Object Close all contexts
if clibIsNull(m2k)
    clib.libm2k.libm2k.context.contextCloseAll();
    m2k = context.m2kOpen();
end

% Calibration
fprintf('Start Calibration TX DAC and ADC\n')
m2k.calibrateADC();
m2k.calibrateDAC();
fprintf('Finished Calibration TX DAC and ADC\n')

% Setup devices
ain = m2k.getAnalogIn();       % UE receives VLC Data
aout = m2k.getAnalogOut();     % gNB sends VLC Data
trig = ain.getTrigger();

% Enables analog input channels
ain.enableChannel(0,true);
ain.setSampleRate(pM2K.pADCSampleRate);

% Define Analog Channel
c1 = analog.ANALOG_IN_CHANNEL.ANALOG_IN_CHANNEL_1;
ain.setRange(c1,-10,10);

% Enable analog output channels
aout.setSampleRate(pM2K.pDACSampleRate);
aout.enableChannel(0, true);
aout.setCyclic(true);


%% Collect analog data
Fs = pM2K.pDACSampleRate;                   % Sampling Rate

fmax = 10e6;                                % Max Frequency Search
fmin = 1e3;                                 % Start Frequency
fpass = 250e3;                              % Frequency change
f=fmin;                                     % Initiate f vector
out = [];                                   % Output buffer
i = 1;
while f < fmax
    m2k.calibrateADC();
    m2k.calibrateDAC();
    % Display current value
    fprintf('Current frequency = %d \n',f);

    % Transmission parameters
    L = 100*round(Fs/f);                        % Buffer length to new tone
    T = 1/Fs;                                   % Sampling Time
    t = (0:L-1)*T;                              % Time vector to new tone

    % Generate new Tone
    S1 = sin(2*pi*f*t);

    % Send Frame
    aout.pushInterleaved(S1,1);

    % Reset Analog Input
    ain.stopAcquisition();

    % Adjuste Sample Rate ADC x DAC
    [P1, Q1] = rat(pM2K.pADCSampleRate/(pM2K.pDACSampleRate/2));
    
    % Get samples from ADALM 2000
    data = ain.getSamplesInterleaved_matlab(2*L);
    data = data.double;

    % Resample to recover initial signal
    data = resample(data,Q1,P1,1);

    % Remove DC Level to only observe the tone energy
    data = data - mean(data);
    
    % Adjuste samples length due to resample process
    L3 = length(data);
    fvec = linspace(0,Fs/2,L3/2);
    t = linspace(0,T*L3,L3);
    
    aout.cancelBuffer();
    pause(0.1);
    % Calculate spectrum
    Y = fft(data);
    P2 = abs(Y/(L3/2));
    P1 = P2(1:L3/2);
    % Partial plot results for each iteration
    % Time Domain Plot
    subplot(2,1,1);
    plot(t,data);
    xlabel('Time (s)')
    ylabel('u.A');
    title('Time Domain Signal Tone')
    
    % Frequency Domain Plot
    subplot(2,1,2);
    plot(fvec,P1);
    xlabel('Frequency (Hz)')
    ylabel('Maginitude u.A');
    title('Frequency Domain Magnitude Spectrum')
    
    % Validate measurements - When DAC slips the spectrum is NaN number
    if max(abs(data)) < 3 && ~isnan(max(abs(P1)))
        out = [out max(abs(data))];
        %out = [out max(abs(P1))];
        [val,idx]=min(abs(fvec-f));
        f = f + fpass;
    end
end

% Normalize received magnitudes
out = out/max(abs(out));

% Convert to Logarihtimic scale
outLog = 10*log(out);

% Plot results
figure();
fvec = [fmin:fpass:fmax];
plot(fvec/1e6,outLog);
ylim([-80,10]);
xlabel('Frequency MHz');
ylabel('Normalized u.A (dB)');
title('ADALM2000 Frequency Response');
grid;

clib.libm2k.libm2k.context.contextCloseAll();

clear m2k
save('m2kFreqResponseBiasTeeNovo','out','outLog','fvec');