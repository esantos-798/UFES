function h_est = toeplitz(M, N, u, y)
    % Monta a matriz de convolucao (Toeplitz) para k = M+1 : N
    % Cada linha k contem [u(k), u(k-1), ..., u(k-M)]
    U_mat = zeros(N - M, M + 1);
    for k = (M+1):N
        U_mat(k - M, :) = u(k:-1:k-M)';
    end
 
    % Vetores de saida correspondentes (a partir da amostra M+1)
    y_rec = y(M+1:N);
 
    % Resolve por minimos quadrados (backslash do Octave ja faz MQ p/ sistema retangular)
    h_est = U_mat \ y_rec ;