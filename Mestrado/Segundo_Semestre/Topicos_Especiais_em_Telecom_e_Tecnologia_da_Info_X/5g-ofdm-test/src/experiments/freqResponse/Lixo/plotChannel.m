function plotChannel(hest,fc,scs)
%PLOTCHANNEL Summary of this function goes here
%   Detailed explanation goes here

% Plot 3D channel
figure();
h=flip(hest,1);
mesh(abs(h(:,:,1,1)));
title('Channel Estimate');
xlabel('OFDM Symbol');
ylabel("Subcarrier");
zlabel("Magnitude");

% Plot 2D Average Channel
figure();
H = mean(hest,2);
H = H/max(abs(H));
H = flip(H);

n = length(H);
f = linspace(fc - (n/2)*scs,fc + (n/2)*scs,n)/1e6;
plot(f,20*log10(abs(H)));
ylim([-7,0]);
grid;
xlabel('Frequency (MHz)')
ylabel('Normalized Amplitude Magnitude');
title('Average Channel Response Frequency');
end

