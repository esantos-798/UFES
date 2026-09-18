function H_est = fourier(u, y)
    % Monta a matriz de convolucao (Toeplitz) para k = M+1 : N
    % Cada linha k contem [u(k), u(k-1), ..., u(k-M)]
    U_f = fft(u);
    Y_f = fft(y);

    % Resolve por minimos quadrados (backslash do Octave ja faz MQ p/ sistema retangular)
    H_est = Y_f ./ U_f ;