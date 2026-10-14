import clib.libm2k.libm2k.*

% Close all Instances
clib.libm2k.libm2k.context.contextCloseAll();

% Open a TX instance
i = 0;
txM2k = context.m2kOpen("ip:192.168.70.62");
while clibIsNull(txM2k) || i < 10
    txM2k = context.m2kOpen("ip:192.168.70.62");
    i = i + 1;
end

% Open a RX instance
i = 0;
rxM2k = context.m2kOpen("ip:192.168.70.42");
while clibIsNull(rxM2k) || i < 10
    rxM2k = context.m2kOpen("ip:192.168.70.42");
    i = i + 1;
end


% Check Opened port
if ~clibIsNull(txM2k) 
    fprintf('Conection Established with ADALM 2000 TX\n');
end

if ~clibIsNull(rxM2k)  
    fprintf('Conection Established with ADALM 2000 RX\n');
end

% Calibration
txM2k.calibrateADC();
txM2k.calibrateDAC();
rxM2k.calibrateADC();
rxM2k.calibrateDAC();

% Setup devices
ain = rxM2k.getAnalogIn();
aout = txM2k.getAnalogOut();
trig = ain.getTrigger();

% Enables analog input channels
ain.enableChannel(0,true);
ain.setSampleRate(pADCSampleRate);

% Define Analog Channel
c1 = analog.ANALOG_IN_CHANNEL.ANALOG_IN_CHANNEL_1;
ain.setRange(c1,-2,2);

% Enable analog output channels
aout.setSampleRate(pDACSampleRate);
aout.enableChannel(0, true);
aout.setCyclic(true);