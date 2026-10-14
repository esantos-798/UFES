function updateFreqDashboard(DLTimeRx,ULTimeRx,TsDL,TsUL)
%UPDATEFREQDASHBOARD Summary of this function goes here
%   Detailed explanation goes here
% Calculate Downlink RX Spectrum
subplot(3,4,5);
[Sinal_ff,~,f,~] = FFT_pot2(DLTimeRx.',TsDL);
espectroAbs = mag2db(abs(Sinal_ff));
plot(f,fftshift(espectroAbs));
title('Downlink RX Spectrum');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (Hz)');
ylim([-200,-100]);
xlim([-10e6,10e6]);

% Calculate Downlink RX Spectrum
subplot(3,4,6);
[Sinal_ff,~,f,~] = FFT_pot2(ULTimeRx.',TsUL);
espectroAbs = mag2db(abs(Sinal_ff));
plot(f,fftshift(espectroAbs));
title('Uplink RX Spectrum');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (Hz)');
ylim([-200,-100]);
xlim([-4.5e6,4.5e6]);
end

