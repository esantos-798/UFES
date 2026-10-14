clear all
close all
clc;

% Add function paths
addpath('../../adalm2000');
addpath('../../adalm-pluto');
addpath('../../etc');
addpath('../../filters');
addpath('../../newradio');
addpath('../VLCTestBed/Results2');

% Get DCI information from 5G Standard
%DCItypes = [1,5,6,11,12,20];
DCItypes = [20];

% Set the Bias DC Current
Idc = "750mA";
distance = '150cm';
evmDLmin = 4.6;
evmDLmax = 5.3;

% Set Channels
%uplinkCH = 'adalm-pluto';         % Set adalm-pluto or software
uplinkCH = 'software';
%downlinkCH = 'adalm2000';       % Set adalm2000 or software
downlinkCH = 'software';

% Loop for modulation parameters variations
for ii=1:length(DCItypes)
    dciCurrentModulation = DCItypes(ii);
    [Qm,R,SE] = getMCSLookupTable(dciCurrentModulation);
    % Start test
    main_experimental;
    
    % Save in results folder
    strData = 'VLCTest_' + Idc + '_downlink-' + downlinkCH+ '_uplink-'+uplinkCH +'_'+ Qm + '_CR-' + R*1024+'-1024_'+distance;
    fileNameData = '..\VLCTestBed\\Results2\' + strData + '.mat';
    save(fileNameData);
    fileNameFig = '..\VLCTestBed\\Results2\' + strData + '.fig';
    savefig(figure(1),fileNameFig);

    close all;
    clc;
end