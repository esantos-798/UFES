clear all;
close all
clc;

% Add function paths
addpath('../etc');

% Load data
load plutoTxFilter;
load waveform;
waveform = waveform/(max(abs(waveform)));


% Create Interpolator Instance
firinterp = dsp.FIRInterpolator(2);

% Apply First Interpolator
y = firinterp(waveform);

% Apply Designed LTE FIR Filter 5 MHZ
y2 = filter(filtTxConfig{4},1,y);

% Apply Next Two Interpolators HB
y3 = firinterp(y2);
y4 = firinterp(y3);
y5 = firinterp(y4);

figure();
subplot(4,1,1);
[Sinal_ff,~,f,df] = FFT_pot2(waveform',1/(7.68e6));
espectroAbs = mag2db(abs(Sinal_ff));
espectroAbs = espectroAbs + 100;
plot(f/1e6,fftshift(espectroAbs));
grid on;
title('a.1) Baseband Spectrum Fs = 7.68 MSPS');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (MHz)');


subplot(4,1,2);
[Sinal_ff,~,f,df] = FFT_pot2(y2.',1/(15.36e6));
espectroAbs = mag2db(abs(Sinal_ff));
plot(f/1e6,fftshift(espectroAbs));
grid on;
title('a.2) Baseband Spectrum after FIR Fs = 15.36 MSPS');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (MHz)');

subplot(4,1,3);
[Sinal_ff,~,f,df] = FFT_pot2(y5.',1/(122.38e6));
espectroAbs = mag2db(abs(Sinal_ff));
plot(f/1e6,fftshift(espectroAbs));
grid on;
title('a.3) Baseband Spectrum after FIR and HB Fs = 122.38 MSPS');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (MHz)');

% Modulation QAM
Fspassband = 24e9;
Fc = 2.4e9;
[P1bb, Q1bb] = rat(Fspassband/122.38e6);

waveform_up = resample(y5,P1bb,Q1bb);
waveform_pb = modulate(real(waveform_up),...
        Fc,Fspassband,'qam',imag(waveform_up));

subplot(4,1,4);
[Sinal_ff,~,f,df] = FFT_pot2(waveform_pb.',1/(Fspassband));
espectroAbs = mag2db(abs(Sinal_ff));
plot(f/1e9,fftshift(espectroAbs));
grid on;
title('a.4) Passband Spectrum Fc = 2.4 GHz');
ylabel('PSD (dB/Hz)');
xlabel('Frequency (GHz)');
xlim([2.3,2.5]);

