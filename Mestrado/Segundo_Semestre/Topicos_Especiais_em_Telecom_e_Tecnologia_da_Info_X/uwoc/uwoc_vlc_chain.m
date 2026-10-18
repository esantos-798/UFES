function [rxWaveform, d] = uwoc_vlc_chain(waveform, Fs_nr, p)
    % uwoc_vlc_chain Substitui o miolo do canal b2b por DUC -> LED -> agua -> fotodetector -> DDC
    %
    %   [rxWaveform, d] = uwoc_vlc_chain(waveform, cInfo.SampleRate, p)
    %
    %   waveform : banda base NR complexa (apos a normalizacao do main)
    %   Fs_nr    : taxa do NR (cInfo.SampleRate)
    %   p        : struct PARCIAL com campos de uwoc_defaults() a sobrescrever (opcional)
    %
    % Pipeline: upsample -> modulate 'qam' (sinal real em fc) -> escala para mA
    %   (pico = I_mod_A) -> led_intensity -> uwoc_apply -> demod 'qam' -> downsample.
    % Usa modulate/demod reais quando existem (MATLAB); senao custom_modulate/custom_demod.
    % rxWaveform sai como COLUNA, ainda sem normalizar (o main normaliza e remove DC depois).
    %
    % d: diagnosticos (waveform_pb, corrente, Lum, h, fracao de clipping, indice de modulacao...).

    if nargin < 3 || isempty(p), p = struct(); end
    pd = uwoc_defaults();
    fn = fieldnames(p);
    for i = 1:numel(fn)
        pd.(fn{i}) = p.(fn{i});
    end
    p = pd;

    hasModDemod = exist('modulate', 'file') > 0 && exist('demod', 'file') > 0;

    waveform = waveform(:);
    [P, Q] = rat(p.Fs_dac / Fs_nr);

    % DUC
    wup = resample(waveform, P, Q);
    if hasModDemod
        pb = modulate(real(wup), p.fc, p.Fs_dac, 'qam', imag(wup));
    else
        pb = custom_modulate(real(wup), p.fc, p.Fs_dac, 'qam', imag(wup));
    end
    pb = pb(:);

    % Sinal -> corrente do LED
    x_mA = p.I_mod_A * pb / max(abs(pb));
    [Lum, I] = led_intensity(p.I_bias, p.alpha, x_mA, p.Lum_max, p.zeta, p.k);

    % SNR: converte a referencia pedida para SNR eletrica (Fs_dac)
    G_dB = 10*log10(p.Fs_dac / (2 * p.K * p.SCS_Hz));       % eletrica -> subportadoras uteis
    switch lower(p.SNR_ref)
        case 'electrical'
            snr_elec = p.SNR_dB;
        case 'rx'
            snr_elec = p.SNR_dB - G_dB;
        case 'software'
            Nfft_nr = Fs_nr / p.SCS_Hz;
            Ps_w = mean(abs(waveform).^2);                   % potencia REAL do waveform normalizado
            snr_rx = p.SNR_dB + 10*log10(Ps_w * Nfft_nr^2 / p.K);
            snr_elec = snr_rx - G_dB;
        otherwise
            error('uwoc_vlc_chain:SNR_ref', 'SNR_ref deve ser ''electrical'', ''rx'' ou ''software''.');
    end

    % Canal + fotodetector + ruido
    h = uwoc_dgf_impulse(p.C, p.Fs_dac, p.t_window_ns, p.cirMethod);
    [rxElec, dch] = uwoc_apply(Lum, h, p.R, p.pdLaw, snr_elec);

    % DDC
    if hasModDemod
        [ri, rq] = demod(rxElec, p.fc, p.Fs_dac, 'qam');
    else
        [ri, rq] = custom_demod(rxElec, p.fc, p.Fs_dac, 'qam');
    end
    rxWaveform_up = ri(:) + 1i*rq(:);
    rxWaveform = resample(rxWaveform_up, Q, P);
    rxWaveform = rxWaveform(:);

    % Diagnosticos
    d.waveform_up = wup;            % banda base a Fs_dac, antes do DUC (para os graficos do main)
    d.rxWaveform_up = rxWaveform_up; % banda base a Fs_dac, apos o DDC
    d.waveform_pb = pb;
    d.current_mA  = I;
    d.Lum         = Lum;
    d.h           = h;
    d.rx_elec     = rxElec;
    d.clipFrac    = mean(p.I_bias + p.alpha*x_mA <= 0);
    d.I_min = min(I);  d.I_max = max(I);
    d.mod_index_rms = p.alpha * sqrt(mean(x_mA.^2)) / p.I_bias;
    d.papr_dB = 10*log10(max(pb.^2) / mean(pb.^2));
    d.params = p;
    d.SNR_elec_eff = snr_elec;   % SNR eletrica efetivamente aplicada
    d.SNR_rx_eff   = snr_elec + G_dB;   % SNR equivalente nas subportadoras uteis
    d.Ps = dch.Ps;  d.Pn = dch.Pn;
end
