function [Qm,R,SE] = getMCSLookupTable(idxMCS)
%GETMCSLOOKUPTABLE Get modulation configuration from table 5.1.3.1-1
% MCS index table 1 for PDSCH
% Input Parameters
% idxMCS - MCS (Modulation Code Scheme) index
% Output Parameters
% Qm - Modulation Order
% R - Code Rate
% SE - spectral efficiency

% Modulation order
Qmcol = ["QPSK";"QPSK";"QPSK";"QPSK";"QPSK";"16QAM";"16QAM";...
    "16QAM";"16QAM";"16QAM";"16QAM";"64QAM";"64QAM";"64QAM";"64QAM";...
    "64QAM";"64QAM";"64QAM";"64QAM";"64QAM";"256QAM";"256QAM";"256QAM";...
    "256QAM";"256QAM";"256QAM";"256QAM";"256QAM";"QPSK";...
    "16QAM";"64QAM";"256QAM"];

% Code rate
Rcol = [120;193;308;449;602;378;434;490;553;616;...
    658;466;517;567;616;666;719;772;822;873;682.5;...
    711;754;797;841;885;916.5;948;-1;-1;-1;-1];

Rcol = Rcol/1024;

% Spectral Efficiency
SEcol = [0.2344;0.3770;0.6016;0.8770;1.1758;...
    1.4766;1.6953;1.9141;2.1602;2.4063;2.5703;...
    2.7305;3.0293;3.3223;3.6094;3.9023;4.2129;4.5234;...
    4.8164;5.1152;5.3320;5.5547;5.8906;6.2266;...
    6.5703;6.9141;7.1602;7.4063;-1;-1;-1;-1];

% Crate table
MCSTable = table(Qmcol,Rcol,SEcol);

% Get MCS information
aux = MCSTable(idxMCS,:);
Qm = aux{1,1};
R = aux{1,2};
SE = aux{1,3};
end

