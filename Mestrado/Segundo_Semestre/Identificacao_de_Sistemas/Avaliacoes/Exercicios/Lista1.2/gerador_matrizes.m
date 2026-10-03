function [A, B, C, D, Ad, Bd, Cd, Dd] = gerador_matrizes(dados, Ts)

    ra = dados(1);
    la = dados(2);
    kb = dados(3);
    jm = dados(4);
    k  = dados(5);
    b  = dados(6);

    % Matrizes do espaço de estados[cite: 1]
    A = [-ra/la 0 -kb/la; 0 0 1; k/jm 0 -b/jm];
    B = [1/la; 0; 0];
    C = [0 0 1];
    D = [0];

    % Sistema Contínuo e Discretização[cite: 1].
    SYSC = ss(A, B, C, D);
    SYSd = c2d(SYSC, Ts, 'zoh');

    Ad = SYSd.A;
    Bd = SYSd.B;
    Cd = SYSd.C;
    Dd = SYSd.D;
