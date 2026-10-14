function grid = custom_nrOFDMDemodulate(carrier, waveform)
    % custom_nrOFDMDemodulate Emula a demodulação OFDM 5G
    %
    % Recupera a grade de recursos original, removendo o Prefixo Cíclico 
    % e aplicando a Transformada Rápida de Fourier (FFT).

    info = custom_nrOFDMInfo(carrier);
    N_fft = info.Nfft;
    cp_len = round(N_fft * (144 / 2048));
    sym_len = N_fft + cp_len;
    
    % Calcula o número de símbolos que chegaram por inteiro
    N_symbols = floor(length(waveform) / sym_len);
    
    % Trunca o vetor para ignorar sobras e recria a matriz
    sync_wave = waveform(1 : N_symbols * sym_len);
    rx_syms_cp = reshape(sync_wave, sym_len, N_symbols);
    
    % Remove o Prefixo Cíclico (pega as amostras finais do bloco)
    rx_syms = rx_syms_cp(cp_len+1:end, :);
    
    % Aplica a FFT e normaliza a energia
    rx_fft = fft(rx_syms, N_fft) / sqrt(N_fft);
    
    % Desfaz o shift, voltando o espectro para o formato centrado no DC
    rx_fft_shifted = fftshift(rx_fft, 1);
    
    % Extrai apenas a banda de interesse (descartando bandas de guarda)
    N_active = carrier.NSizeGrid * 12;
    start_idx = floor(N_fft/2 - N_active/2) + 1;
    end_idx = start_idx + N_active - 1;
    
    grid = rx_fft_shifted(start_idx:end_idx, :);
end
