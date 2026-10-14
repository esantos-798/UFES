function [rx, d] = uwoc_apply(Lum, h, R, pdLaw, SNR_dB)
    % uwoc_apply Canal optico: convolucao com a CIR, fotodeteccao, bloqueio de DC e ruido
    %
    %   Lum     saida optica do LED (com DC), vetor
    %   h       resposta ao impulso (soma 1)
    %   pdLaw   'linear' : i = R*y        |  'square' : i = R*|y|^2  (como SimLED_conv.m)
    %   SNR_dB  SNR eletrico do sinal AC (medido), ruido branco gaussiano real; Inf = sem ruido
    %
    % A convolucao e CAUSAL e inicia em regime (bordas preenchidas com Lum(1)), logo nao
    % introduz o atraso de meia janela do conv(...,'same').

    Lum = Lum(:);  h = h(:);
    nh = numel(h);
    yfull = filter(h, 1, [repmat(Lum(1), nh-1, 1); Lum]);
    y = yfull(nh:end);

    switch lower(pdLaw)
        case 'linear', ipd = R * y;
        case 'square', ipd = R * abs(y).^2;
        otherwise
            error('uwoc_apply:pdLaw', 'pdLaw deve ser ''linear'' ou ''square''.');
    end

    ac = ipd - mean(ipd);                       % DC block
    Ps = mean(ac.^2);
    if isinf(SNR_dB)
        Pn = 0;  rx = ac;
    else
        Pn = Ps / 10^(SNR_dB/10);
        rx = ac + sqrt(Pn) * randn(size(ac));
    end
    d.Ps = Ps;  d.Pn = Pn;  d.ipd_dc = mean(ipd);
end
