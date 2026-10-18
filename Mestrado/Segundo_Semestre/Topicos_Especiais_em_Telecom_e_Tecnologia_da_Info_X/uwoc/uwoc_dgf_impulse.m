function [h, t_ns] = uwoc_dgf_impulse(C, Fs, t_window_ns, method)
    % uwoc_dgf_impulse Resposta ao impulso Double Gamma, amostrada a Fs e normalizada (sum(h)=1)
    %
    %   h(t) = C1*t*exp(-C2*t) + C3*t*exp(-C4*t),   t em ns,  0 <= t <= t_window_ns
    %
    % method:
    %   'sample'    amostra h em t = 0:Ts:T (igual ao SimLED_conv.m). Em Fs baixo (75 MS/s)
    %               h(0) = 0 e o resultado vira um ATRASO espurio de 1 amostra.
    %   'integrate' (padrao) integra h dentro de cada intervalo de amostra [n*Ts,(n+1)*Ts)
    %               (grade fina com >= 16 pontos por amostra). A 75 MS/s a CIR de 16 ns
    %               cai quase toda na amostra 0 -> h ~ delta, sem atraso.
    % h e retornado como vetor LINHA.

    if nargin < 4 || isempty(method), method = 'integrate'; end
    Ts = 1 / Fs;
    dgf = @(t) C(1) .* t .* exp(-C(2) .* t) + C(3) .* t .* exp(-C(4) .* t);

    switch lower(method)
        case 'sample'
            t_ns = (0:Ts:t_window_ns*1e-9) * 1e9;
            h = dgf(t_ns);
        case 'integrate'
            nBins = max(1, ceil(t_window_ns*1e-9 * Fs - 1e-9));
            M = max(16, ceil(1e9 / Fs));                       % pontos finos por amostra
            tf = ((0:nBins*M-1) + 0.5) * (Ts / M) * 1e9;       % pontos medios [ns]
            hf = dgf(tf);
            hf(tf > t_window_ns) = 0;                          % respeita a janela
            h = sum(reshape(hf, M, nBins), 1);
            t_ns = (0:nBins-1) * Ts * 1e9;
        otherwise
            error('uwoc_dgf_impulse:Method', 'method deve ser ''sample'' ou ''integrate''.');
    end
    h = h / sum(h);
end
