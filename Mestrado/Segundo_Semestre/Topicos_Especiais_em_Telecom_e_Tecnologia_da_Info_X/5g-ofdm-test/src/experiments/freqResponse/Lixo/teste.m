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
pM2K.pDACSampleRate = 150e6;              % DAC Available Sample Rate
pM2K.pADCSampleRate = 200e6;             % ADC Available Sample Rate

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
aout = m2k.getAnalogOut();    % gNB sends VLC Data
trig = ain.getTrigger();

% Enables analog input channels
ain.enableChannel(0,true);
ain.setSampleRate(pM2K.pADCSampleRate);

% Define Analog Channel
c1 = analog.ANALOG_IN_CHANNEL.ANALOG_IN_CHANNEL_1;
ain.setRange(c1,-2,2);

% Enable analog output channels
aout.setSampleRate(pM2K.pDACSampleRate);
aout.enableChannel(0, true);
aout.setCyclic(true);

%% Collect analog data
Fs = pM2K.pDACSampleRate;                   % Sampling Rate
fc=3e6;

% Transmission parameters
L = round(Fs/fc);
T = 1/Fs;                                   % Sampling Time
t = (0:L-1)*T;                              % Time vector

% Generate new Tone
S1 = sin(2*pi*fc*t);

% Send Frame
aout.pushInterleaved(S1,1);

% Reset Analog Input
ain.stopAcquisition();

Ladc = 2048;
Fsadc = pM2K.pADCSampleRate;
Tadc = 1/Fsadc;
tadc = (0:Ladc-1)*Tadc;
fadc = Fsadc*(0:(Ladc/2))/Ladc;

data = ain.getSamplesInterleaved_matlab(Ladc);
data = data.double;


Y = fft(data);
P2 = abs(Y/(Ladc/2));
P1 = P2(1:Ladc/2+1);

subplot(2,1,1);
plot(tadc,data);
subplot(2,1,2);
plot(fadc,P1);
xlim([0 15e6]);
clib.libm2k.libm2k.context.contextCloseAll();

clear m2k