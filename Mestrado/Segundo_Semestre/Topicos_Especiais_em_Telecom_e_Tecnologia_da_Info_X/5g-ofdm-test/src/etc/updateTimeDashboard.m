function updateTimeDashboard(DLTimeTx,DLTimeRx,ULTimeTx,ULTimeRx,Ts)

% Start Figure
figure(1);
set(gcf, 'Position', [1 41 1280 600.333])
            
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

end