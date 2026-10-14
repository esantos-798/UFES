function [Lum, I] = led_intensity(I_bias, alpha, x, Lum_max, zeta, k)
    % led_intensity Modelo estatico do LED (SimLED_conv.m)
    %
    %   I   = I_bias + alpha * x            (x = sinal modulante em mA; I limitado a >= 0)
    %   Lum = I ./ (zeta + (I/Lum_max).^(2k)).^(1/(2k))
    %
    % Retorna a saida optica Lum e a corrente efetiva I (apos o limite em 0).
    I = I_bias + alpha * x;
    I = max(I, 0);
    Lum = I ./ (zeta + (I / Lum_max).^(2*k)).^(1/(2*k));
end
