function updateDashboard(DLTimeTx,DLTimeRx,ULTimeTx,ULTimeRx,...
    DLcns,DLcnsEq,ULcns,ULcnsEq,DLgrid,...
    ULgrid,berDL,berUL,evmDL,evmUL,retryCount,iteration,Ts)

% Start Figure
figure(1);
set(gcf, 'Position', [1 41 1280 607.3333])
            
% Create auxiliar time vector (5G NR always have 10ms per Frame)


% Downlink Time Signal Plot
subplot(3,4,1)
t = [0:Ts:Ts*(length(DLTimeTx)-1)];
plot(t, real(DLTimeTx), 'b-', 'linewidth', 1), hold on;
t = [0:Ts:Ts*(length(DLTimeRx)-1)];
plot(t, real(DLTimeRx), 'r--', 'linewidth', 1)
title('Downlink Time Domain')
xlabel('Time (s)')
ylabel('Amplitude (V)')
legend('Transmitted','Received')
hold off

% Uplink Time Signal Plot
subplot(3,4,2)
t = [0:Ts:Ts*(length(ULTimeTx)-1)];
plot(t, real(ULTimeTx), 'b-', 'linewidth', 1), hold on;
t = [0:Ts:Ts*(length(ULTimeRx)-1)];
plot(t, real(ULTimeRx), 'r--', 'linewidth', 1)
title('Uplink Time Domain')
xlabel('Time (s)')
ylabel('Amplitude (V)')
legend('Transmitted','Received')
hold off

% Calculate Downlink RX Spectrum
subplot(3,4,5);
[Sinal_ff,~,f,~] = FFT_pot2(DLTimeRx.',Ts);
espectroAbs = mag2db(abs(Sinal_ff));
plot(f,fftshift(espectroAbs));
title('Downlink RX Spectrum');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (Hz)');
ylim([-200,-100]);

% Calculate Downlink RX Spectrum
subplot(3,4,6);
[Sinal_ff,~,f,~] = FFT_pot2(ULTimeRx.',Ts);
espectroAbs = mag2db(abs(Sinal_ff));
plot(f,fftshift(espectroAbs));
title('Uplink RX Spectrum');
ylabel('PSD (dB/Hz)');
xlabel('Frequêncy (Hz)');
ylim([-200,-100]);

% Plot Downlink Constellation not Equalized yet
subplot(3,4,7);
plot(real(DLcns),imag(DLcns),'b.');
%a = max(max(real(DLcns)));
%axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
title ('PDSCH RX constellation');
xlabel('Real'), ylabel('Imag');


% Plot Downlink Constellation Equalized
subplot(3,4,8);
plot(real(DLcnsEq),imag(DLcnsEq),'b.');
%axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
title ('PDSCH RX constellation Equalized');
xlabel('Real'), ylabel('Imag');
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);

% Plot Downlink Resource Grid
subplot(3,4,9);
imagesc(abs(DLgrid));
axis xy;
xlabel('OFDM symbol');
ylabel('Subcarrier');
title('Downlink Resource Grid')

% Plot Uplink Resource Grid
subplot(3,4,10);
imagesc(abs(ULgrid));
axis xy;
xlabel('OFDM symbol');
ylabel('Subcarrier');
title('Uplink Resource Grid')

% Plot Downlink Constellation not Equalized yet
subplot(3,4,11);
plot(real(ULcns),imag(ULcns),'r.');
%a = max(max(real(ULcns)));
%axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
title ('PUSCH RX constellation');
xlabel('Real'), ylabel('Imag');


% Plot Downlink Constellation Equalized
subplot(3,4,12);
plot(real(ULcnsEq),imag(ULcnsEq),'r.');
%axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
title ('PUSCH RX constellation Equalized');
xlabel('Real'), ylabel('Imag');
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);
% Annotations

pvert = 0.85;
delete(findall(gcf,'type','annotation'))
delt_pvert = 0.05;

% BER Downlink
pos = [0.525 pvert-0*delt_pvert 1 0.08];
str = ['BER Downlink = ', num2str(berDL)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% BER Uplink 
pos = [0.525 pvert-1*delt_pvert 1 0.08];
str = ['BER Uplink = ', num2str(berUL)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% EVM Downlink
pos = [0.525 pvert-2*delt_pvert 1 0.08];
str = ['EVM Downlink = ', num2str(evmDL), ' %'];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% EVM Uplink
pos = [0.525 pvert-3*delt_pvert 1 0.08];
str = ['EVM Uplink = ', num2str(evmUL), ' %'];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% Iteration Counter
pos = [0.7 pvert-0*delt_pvert 1 0.08];
str = ['Iteration = ', num2str(iteration)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% Retransmission Counter
pos = [0.7 pvert-1*delt_pvert 1 0.08];
str = ['Retransmissions = ', num2str(retryCount)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

end

