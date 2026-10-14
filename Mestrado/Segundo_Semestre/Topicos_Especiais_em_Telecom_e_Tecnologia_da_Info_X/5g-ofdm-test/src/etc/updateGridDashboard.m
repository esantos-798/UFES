function updateGridDashboard(DLgrid,ULgrid)
%UPDATEGRIDDASHBOARD Summary of this function goes here
%   Detailed explanation goes here
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
end

