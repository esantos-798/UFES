function updateCnstDashboard(DLcns,DLcnsEq,ULcns,ULcnsEq)
%UPDATECONSTELLATION Summary of this function goes here
%   Detailed explanation goes here
% Plot Downlink Constellation not Equalized yet
% subplot(3,4,7);
% plot(real(DLcns),imag(DLcns),'b.');
% %a = max(max(real(DLcns)));
% %axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
% title ('PDSCH RX constellation');
% xlabel('Real'), ylabel('Imag');


% Plot Downlink Constellation Equalized
subplot(3,4,8);
plot(real(DLcnsEq),imag(DLcnsEq),'b.');
%axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
title ('PDSCH RX constellation Equalized');
xlabel('Real'), ylabel('Imag');
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);

% % Plot Downlink Constellation not Equalized yet
% subplot(3,4,11);
% plot(real(ULcns),imag(ULcns),'r.');
% %a = max(max(real(ULcns)));
% %axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
% title ('PUSCH RX constellation');
% xlabel('Real'), ylabel('Imag');


% Plot Downlink Constellation Equalized
subplot(3,4,12);
plot(real(ULcnsEq),imag(ULcnsEq),'r.');
%axis ([-a-0.1 a+0.1 -a-0.1 a+0.1]);
title ('PUSCH RX constellation Equalized');
xlabel('Real'), ylabel('Imag');
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);
end

