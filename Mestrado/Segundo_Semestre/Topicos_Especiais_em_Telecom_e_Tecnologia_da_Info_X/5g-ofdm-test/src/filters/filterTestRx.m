clear all;
close all
clc;

% Add function paths
addpath('../etc');

% Load data
load plutoTxFilter;
load waveform_pb;
waveform_pb = waveform_pb/(max(abs(waveform_pb)));

% Apply Noise to simulate rx
snr = 5;
waveform_pb = awgn(waveform_pb,snr);

% Modulation QAM
Fspassband = 24e9;
Fc = 2.4e9;
[P1bb, Q1bb] = rat(Fspassband/122.38e6);

% Demodulate
[rxReal,rxImag] = demod(waveform_pb,...
        Fc,Fspassband,'qam');
rxWaveform = rxReal + 1i.*rxImag;

% Downsample
y5 = resample(rxWaveform,Q1bb,P1bb);

% Create Interpolator Instance
firdecim = dsp.FIRDecimator(2);

y4 = firdecim(y5);
y3 = firdecim(y4);
y2 = firdecim(y3);

y1 = filter(filtTxConfig{4},1,y2);

y = firdecim(y1);

figure();
subplot(4,1,1);
[Sinal_ff,~,f,df] = FFT_pot2(waveform_pb',1/Fspassband);
espectroAbs = mag2db(abs(Sinal_ff));
espectroAbs = espectroAbs + 100;
plot(f/1e9,fftshift(espectroAbs));
grid on;
title('b.4) Passband Spectrum Fc = 2.4GHz');
ylabel('PSD (dB/Hz)');
xlabel('Frequency (GHz)');
xlim([2.3 2.5]);

subplot(4,1,2);
[Sinal_ff,~,f,df] = FFT_pot2(y5',1/122.38e6);
espectroAbs = mag2db(abs(Sinal_ff));
espectroAbs = espectroAbs + 100;
plot(f/1e6,fftshift(espectroAbs));
grid on;
title('b.3) Baseband Spectrum before HB Fs = 122.38 MSPS');
ylabel('PSD (dB/Hz)');
xlabel('Frequency (MHz)');

subplot(4,1,3);
[Sinal_ff,~,f,df] = FFT_pot2(y2',1/15.36e6);
espectroAbs = mag2db(abs(Sinal_ff));
espectroAbs = espectroAbs + 100;
plot(f/1e6,fftshift(espectroAbs));
grid on;
title('b.2) Baseband Spectrum before FIR Fs = 15.36 MSPS');
ylabel('PSD (dB/Hz)');
xlabel('Frequency (MHz)');

subplot(4,1,4);
[Sinal_ff,~,f,df] = FFT_pot2(y',1/7.68e6);
espectroAbs = mag2db(abs(Sinal_ff));
espectroAbs = espectroAbs;
plot(f/1e6,fftshift(espectroAbs));
grid on;
title('b.1) Baseband Spectrum received Fs = 7.68 MSPS');
ylabel('PSD (dB/Hz)');
xlabel('Frequency (MHz)');



