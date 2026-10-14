function updateChannelDashboard(DLChannel,ULChannel)
%UPDATECHANNELDASHBOARD Summary of this function goes here

%   Detailed explanation goes here
% % Plot Downlink Constellation Equalized
subplot(3,4,7);
mesh(abs(DLChannel(:,:,1,1)));
title('Channel Estimate Downlink');
xlabel('OFDM Symbol');
ylabel("Subcarrier");
zlabel("Magnitude");
zlim([0 mean(mean(abs(DLChannel)))+1]);

% Plot Downlink Constellation not Equalized yet
subplot(3,4,11);
mesh(abs(ULChannel(:,:,1,1)));
title('Channel Estimated Uplink');
xlabel('OFDM Symbol');
ylabel("Subcarrier");
zlabel("Magnitude");
zlim([0 mean(mean(abs(ULChannel)))+1]);
end

