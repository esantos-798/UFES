import clib.libm2k.libm2k.*

% Open a TX instance
gNBM2k = context.m2kOpen("usb:1.10.5");

% If NULL Object Close all contexts
if clibIsNull(gNBM2k)
    clib.libm2k.libm2k.context.contextCloseAll();
    gNBM2k = context.m2kOpen();
end

% Open a RX Instance
UEM2k = context.m2kOpen("usb:1.9.5");

if clibIsNull(UEM2k)
    clib.libm2k.libm2k.context.contextCloseAll();
    gNBM2k = context.m2kOpen();
    UEM2k = context.m2kOpen();
end

% Check Opened port
if ~isempty(gNBM2k) 
    fprintf('Connection Established with ADALM 2000 TX\n');
end

if ~isempty(UEM2k) 
    fprintf('Connection Established with ADALM 2000 RX\n');
end

% Calibration
fprintf('Start Calibration TX DAC and ADC\n')
gNBM2k.calibrateADC();
gNBM2k.calibrateDAC();
fprintf('Finished Calibration TX DAC and ADC\n')

fprintf('Start Calibration RX DAC and ADC\n')
UEM2k.calibrateADC();
UEM2k.calibrateDAC();
fprintf('Finished Calibration RX DAC and ADC\n')

% Setup devices
ain = UEM2k.getAnalogIn();       % UE receives VLC Data
aout = gNBM2k.getAnalogOut();    % gNB sends VLC Data
trig = ain.getTrigger();

% Enables analog input channels
ain.enableChannel(0,true);
ain.setSampleRate(pM2K.pADCSampleRate);

% Define Analog Channel
c1 = analog.ANALOG_IN_CHANNEL.ANALOG_IN_CHANNEL_1;
ain.setRange(c1,-2,2);
%ain.setRange(c1,-10,10);

% Enable analog output channels
aout.setSampleRate(pM2K.pDACSampleRate);
aout.enableChannel(0, true);
aout.setCyclic(true);