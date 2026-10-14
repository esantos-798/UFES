gNB.NDLRB = 25;               % Bandwidth in number of resource blocks
gNB.SubcarrierSpacing = 15;    % 15, 30, 60, 120, 240, 480 (kHz)
gNB.WaveformType = 'CP-OFDM';  % 'CP-OFDM', 'W-OFDM' or 'F-OFDM'
gNB.CyclicPrefix = 'Normal';   % 'Normal' or 'Extended'
gNB.UseDCSubcarrier = 'Off';   % 'On' or 'Off'
gNB.NCellID = 102;               % Cell identity

gNB.SSBurst.BurstType = 'CaseC';   
gNB.SSBurst.SubcarrierOffset = 0;

% Bitmap indicating blocks transmitted in the burst
gNB.SSBurst.SSBTransmitted = [1 1 1 1];

% SS burst set periodicity in ms (5, 10, 20, 40, 80, 160)
gNB.SSBurst.SSBPeriodicity = 10;        
 

% Create SS burst waveform and information structure and display a plot
% showing the SS burst content (in the SS burst numerology)
ssburst = gNB.SSBurst;
ssburst.DisplayBurst = true;
[ssbWaveform,~,ssbInfo] = h5gSSBurst(gNB,ssburst);

 

