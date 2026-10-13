% Salve este arquivo exatamente como: awgn.m
function y = awgn_custom(x, snr, varargin)
    % Versão "na unha" da função awgn para Octave Puro
    % x: sinal de entrada (pode ser complexo, como o QAM/OFDM)
    % snr: Relação Sinal-Ruído em dB
    
    % 1. Calcular a potência do sinal de entrada (Pm)
    % Em OFDM/QAM, medimos a energia média dos símbolos
    Pm = mean(abs(x(:)).^2);
    
    % 2. Converter a SNR de dB para escala linear
    snr_linear = 10^(snr / 10);
    
    % 3. Calcular a potência que o ruído deve ter (Pn)
    Pn = Pm / snr_linear;
    
    % 4. Gerar o ruído gaussiano com a potência correta
    % Se o sinal for complexo, precisamos de ruído na parte real e imaginária
    if ~isreal(x)
        % Divisão por sqrt(2) para que a potência total (Real + Imag) seja Pn
        ruido = (randn(size(x)) + 1i * randn(size(x))) * sqrt(Pn / 2);
    else
        ruido = randn(size(x)) * sqrt(Pn);
    end
    
    % 5. Somar o ruído ao sinal original
    y = x + ruido;
end