function [waveform, info] = custom_nrOFDMModulate(carrier, grid)
    % custom_nrOFDMModulate Emula a modulação OFDM 5G (IFFT e Prefixo Cíclico)
    %
    % Entrada:
    % - carrier: Estrutura da portadora
    % - grid: Matriz de Recursos contendo os símbolos QAM (Subportadoras x Símbolos)
    %
    % Saída:
    % - waveform: Vetor serializado no domínio do tempo

    info = custom_nrOFDMInfo(carrier);
    N_fft = info.Nfft;
    
    [N_active, N_symbols] = size(grid);
    
    % Mapeamento dos símbolos na grade FFT
    fft_grid = zeros(N_fft, N_symbols);
    
    % Distribuição simétrica das subportadoras em torno do DC (Frequência Central)
    start_idx = floor(N_fft/2 - N_active/2) + 1;
    end_idx = start_idx + N_active - 1;
    
    fft_grid(start_idx:end_idx, :) = grid;
    
    % IFFT Shift para mover o DC (centro) para o índice 1 antes da Transformada
    fft_grid_shifted = ifftshift(fft_grid, 1);
    
    % Transformada Inversa e normalização de energia
    time_syms = ifft(fft_grid_shifted, N_fft) * sqrt(N_fft);
    
    % Adição do Prefixo Cíclico (CP)
    % No 5G NR (SCS 15kHz), o CP normal baseia-se em 144 amostras para uma FFT de 2048.
    % Mapeamos essa proporção de acordo com o tamanho real da nossa FFT.
    cp_len = round(N_fft * (144 / 2048));
    
    time_syms_cp = [time_syms(end-cp_len+1:end, :); time_syms];
    
    % Serialização final: lê coluna por coluna e converte em vetor único
    waveform = time_syms_cp(:);
end
