import clib.libm2k.libm2k.*

% Open a TX instance
txM2k = context.m2kOpen();

% If NULL Object Close all contexts
if clibIsNull(txM2k)
    clib.libm2k.libm2k.context.contextCloseAll();
    txM2k = context.m2kOpen();
end

% Open a RX Instance
rxM2k = context.m2kOpen();

% Check Opened port
if ~isempty(txM2k) 
    fprintf('Conection Established with ADALM 2000 TX\n');
end

if ~isempty(rxM2k) 
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
ain.setRange(c1,-1,1);

% Enable analog output channels
aout.setSampleRate(pDACSampleRate);
aout.enableChannel(0, true);
aout.setCyclic(true);