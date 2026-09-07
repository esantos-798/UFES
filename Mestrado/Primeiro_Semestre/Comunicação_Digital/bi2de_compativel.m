function dec = bi2de_compativel(bin_matriz)
    % Converte matriz binária de 2 colunas para decimal (0 a 3)
    dec = bin_matriz(:,1)*2 + bin_matriz(:,2);
end