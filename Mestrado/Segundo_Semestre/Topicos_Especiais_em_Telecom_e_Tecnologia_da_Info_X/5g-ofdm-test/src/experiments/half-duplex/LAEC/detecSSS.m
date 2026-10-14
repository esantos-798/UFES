function [NID1,ncellid] = detecSSS(sssRx,NID2,plotConfig)
% Detect Second Synchronization Symbol with cross-correlation 
% with received sss and reference sss signals

% Input Parameters
% sssRX - Second Synchronization Symbol Received
% NID2 - Cell NID2 [0,1,2]
% plotConfig - true or false

% Output Parameters
% NID1 - cell ID NID1 [0:335]
% cellID - cell ID = 3xNID1 + NID2

% Correlate received SSS symbols with each possible SSS sequence
sssEst = zeros(1,336);
for NID1 = 0:335

    ncellid = (3*NID1) + NID2;
    sssRef = nrSSS(ncellid);
    sssEst(NID1+1) = sum(abs(mean(sssRx .* conj(sssRef),1)).^2);
end

% Determine NID1 by finding the strongest correlation
NID1 = find(sssEst==max(sssEst)) - 1;

% Form overall cell identity from estimated NID1 and NID2
ncellid = (3*NID1) + NID2;

% Display Cell ID number
disp([' Cell identity: ' num2str(ncellid)])

% Plot Correlations
if plotConfig
    % Plot SSS correlations
    figure;
    stem(0:335,sssEst,'o');
    title('SSS Correlations (Frequency Domain)');
    xlabel('$N_{ID}^{(1)}$','Interpreter','latex');
    ylabel('Magnitude');
    axis([-1 336 0 max(sssEst)*1.1]);
    
    % Plot selected NID1
    hold on;
    plot(NID1,max(sssEst),'kx','LineWidth',2,'MarkerSize',8);
    legend(["correlations" "$N_{ID}^{(1)}$ = " + num2str(NID1)],'Interpreter','latex');
end
end

