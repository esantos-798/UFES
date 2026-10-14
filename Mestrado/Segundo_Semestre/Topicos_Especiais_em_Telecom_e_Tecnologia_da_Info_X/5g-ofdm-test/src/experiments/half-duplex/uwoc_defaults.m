function p = uwoc_defaults()
    % uwoc_defaults Parametros padrao do enlace VLC/UWOC (Fase 1)
    %
    % LED e fotodetector: valores de SimLED_conv.m. Canal: coeficientes Double
    % Gamma de SimLED_conv.m (t em ns). Qualquer campo pode ser sobrescrito
    % passando uma struct parcial para uwoc_vlc_chain.

    % --- Banda passante (mesmos valores do b2b/adalm2000) ---
    p.fc     = 5e6;      % portadora [Hz]            (pVLCCenterFrequency)
    p.Fs_dac = 75e6;     % taxa do DAC/canal [Hz]    (pDACSampleRate)

    % --- LED (SimLED_conv.m) ---
    p.I_bias  = 800;     % corrente de polarizacao [mA]
    p.I_mod_A = 50;      % pico do sinal modulante ANTES do ganho alpha [mA]
    p.alpha   = 10;      % ganho do sinal modulante
    p.Lum_max = 755;     % parametro de saturacao do modelo
    p.zeta    = 20;      % parametro de saturacao (> 0)
    p.k       = 1.9;     % fator de forma
    % excursao de pico em torno do bias = alpha * I_mod_A (= 500 mA, como no SimLED)

    % --- Fotodetector ---
    p.R     = 0.6;       % responsividade
    p.pdLaw = 'linear';  % 'linear' (fotocorrente ~ potencia optica) | 'square' (replica SimLED_conv.m: R*|y|^2)

    % --- Canal: resposta ao impulso Double Gamma  h(t)=C1 t e^{-C2 t}+C3 t e^{-C4 t}, t em ns ---
    p.C           = [4.777160 3.335616 0.529837 0.799894];
    p.t_window_ns = 16;          % janela da CIR [ns]
    p.cirMethod   = 'integrate'; % 'integrate' (soma por intervalo de amostra) | 'sample' (como no SimLED_conv.m)

    % --- Ruido ---
    % SNR eletrico do sinal AC amostrado a Fs_dac, ruido branco em 0..Fs_dac/2 (Inf = sem ruido).
    % A SNR efetiva na banda do sinal e maior (a banda util e ~4,5 MHz de 37,5 MHz).
    p.SNR_dB  = 30;
    % Referencia da SNR_dB:
    %   'electrical' (padrao) : SNR do sinal AC amostrado a Fs_dac, ruido branco em 0..Fs_dac/2
    %   'rx'                  : SNR medida nas subportadoras uteis apos a demodulacao OFDM
    %                           (SNR_rx = SNR_eletrica + 10*log10(Fs_dac/(2*K*SCS_Hz)))
    %   'software'            : SNR_dB e o pSNRdB do canal 'software'; converte para a MESMA SNR real
    %                           no receptor que o canal 'software' produz (o awgn dele assume potencia
    %                           nominal -10*log10(Nfft) dBW, nao a potencia real do waveform)
    p.SNR_ref = 'electrical';
    p.K       = 300;     % subportadoras uteis (12*NSizeGrid)
    p.SCS_Hz  = 15e3;    % espacamento entre subportadoras [Hz]
end
