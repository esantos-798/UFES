clear all;
close all;
clc;

M = 4;
R = 120;
DR = 3.86 * R/1024 * log2(M);

fprintf("DR = %.2f Mbps, para R = %d e M = %d \n",DR,R,M);

M = 4;
R = 602;
DR = 3.86 * R/1024 * log2(M);

fprintf("DR = %.2f Mbps, para R = %d e M = %d \n",DR,R,M);

M = 16;
R = 378;
DR = 3.86 * R/1024 * log2(M);

fprintf("DR = %.2f Mbps, para R = %d e M = %d \n",DR,R,M);

M = 16;
R = 658;
DR = 3.86 * R/1024 * log2(M);

fprintf("DR = %.2f Mbps, para R = %d e M = %d \n",DR,R,M);

M = 64;
R = 466;
DR = 3.86 * R/1024 * log2(M);

fprintf("DR = %.2f Mbps, para R = %d e M = %d \n",DR,R,M);

M = 64;
R = 873;
DR = 3.86 * R/1024 * log2(M);

fprintf("DR = %.2f Mbps, para R = %d e M = %d \n",DR,R,M);