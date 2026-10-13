function matriz_out = repat_compativel(vetor_coluna, num_reps)
    % Substitui o repmat para compatibilidade estrita
    matriz_out = zeros(length(vetor_coluna), num_reps);
    for col = 1:num_reps
        matriz_out(:, col) = vetor_coluna;
    end
end