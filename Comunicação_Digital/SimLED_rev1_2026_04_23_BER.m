% =========================================================
%  Simulação do modelo de LED — Potência Fotométrica
% =========================================================
%                                         Cristiano Tavares
%                                                Abril/2026
%
%  Equação:
%                        I_LED
%  Lum_LED = ─────────────────────────────────
%            [ ζ + (I_LED / P_max)^2k ]^(1/2k)
%
%  Unidades:
%    I_LED    → corrente elétrica       [mA]
%    Lum      → luminância              [lux]
%    Lum_max  → luminância máxima       [lux]
% =========================================================

clc; clear; close all; close

% ── Parâmetros do modelo ──────────────────────────────────
Lum_max = 755;          % Iluminância máxima  [lux]
zeta  = 20;             % Parâmetro de saturação  (adimensional, > 0)
k     = 1.9;            % Fator de forma (inteiro positivo)
SNR_dB = 30;            % Relação sinal-ruído
R = 0.6;                % Responsatividade (Camporez, et al. 2024)
N_bits  = 300000;       % Número de bits da sequência
I_bias = 800;           % Amplitude da coorente da portadora
I_mod_A = 10;           % Amplitude da corrente modulante [mA]
%                           bit=1 → +I_mod_A  |  bit=0 → -I_mod_A  (NRZ bipolar)
alpha = 10;
Tempo_pulso = 4; % Amostras por bit (inteiro)
SNRmax=12;

% ============================================================
%  CURVA DE LUMINÂNCIA
% ============================================================

% ─── Corrente de entrada ────────────────────────────────────
I_LED = linspace(0, 2500, 500);   % Varredura [mA]

% ─── Função do modelo ─ Cálculod da luminância ──────────────
Lum = I_LED ./ (zeta + (I_LED / Lum_max).^(2*k)).^(1/(2*k));


% =========================================================
%  MODULAÇÃO DO LED  —  I_fin = I_bias + I_mod 
% =========================================================

% Amostras por bit — calculado para acomodar exatamente 2048 bits
% no mesmo comprimento de i_sinal
N_uso = N_bits * Tempo_pulso;               % Amostras efetivamente usadas

% 1) Sequência de bits aleatórios 
rng(42);                                    % Semente — garante reprodutibilidade
bits = randi([0 1], 1, N_bits);             % 2048 bits: 0 ou 1

% 2) Sinal modulante NRZ bipolar → corrente 
niveis = (2*bits - 1) * I_mod_A;            % +I_mod_A ou -I_mod_A
I_mod  = rectpulse(niveis, Tempo_pulso);      % Dá corpo ao pulso

% 3) Sinal final modulado
I_fin = I_bias + alpha*I_mod;

% 4) Potência óptica de saída
Lum_fin = I_fin ./ (zeta + (I_fin / Lum_max).^(2*k)).^(1/(2*k));
 
% 5) Modelagem de fotodetecção
Ir = R*abs(Lum_fin).^2;                     % (Pereira, et al.2015)    
                                            % Ir -> Corrente receptor
% 6) DC Block
Irb = Ir - mean(Ir);                        %retira o valor médio
 
% 7) Aplicando ruído na corrente recebida
SNR_dB = SNR_dB + 10*log10(Tempo_pulso);
Ir_ruido = awgn(Irb, SNR_dB, 'measured');
 

% =========================================================
%  GRÁFICO BER - Bit error rate
% =========================================================

BER = zeros(SNRmax, 1);
snr_vetor = 1:SNRmax;
for SNR=snr_vetor 
    % Aplicando ruído na corrente recebida
    SNR_dB = SNR - 10*log10(Tempo_pulso)+3;    
                               % 10*log10(Tempo_pulso) -> adicionado para compensar a largura do pulso
                               % +3dB -> adicionado pois o pulso é um valor inteiro (sem parte imaginária) 

    Ir_ruido = awgn(Irb, SNR_dB, 'measured');
    
    % Retirando larguras de Ir_ruido
    Ir_ruido_down = intdump(Ir_ruido,Tempo_pulso);
    
    bits_finais = Ir_ruido_down > 0;        %verifica se o valor é positivo ou negativo
    
    %Verifica quantos bits estão errados
    erros = 0;
    for i = 1:length(bits_finais)
        if bits(i) ~= bits_finais(i)
            erros = erros + 1;
        end
    end
    BER(SNR) = erros/length(bits_finais);

end

figure;
semilogy(snr_vetor, BER, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('Desempenho BER vs SNR');
legend('Modelo simulado');


% --- Cálculo da Curva Teórica ---
% Convertendo o vetor de SNR de dB para escala linear
snr_linear = 10.^(snr_vetor / 10);
BER_teorica = 0.5 * erfc(sqrt(snr_linear)); %(Proakis e Salehi, 2008)

BER_teorica_pam = berawgn(snr_vetor,"pam",2);
                                            

% --- Plotagem Comparativa ---
figure;
semilogy(snr_vetor, BER, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 6); % Sua simulação
hold on;
semilogy(snr_vetor, BER_teorica, 'r--', 'LineWidth', 2); % Teoria
grid on;
semilogy(snr_vetor, BER_teorica_pam, 'g*', 'LineWidth', 2); % Teoria
grid on;

xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('Desempenho de Erro: Simulação LED vs. Teoria NRZ');
legend('Simulação (Modelo LED)', 'Teoria (NRZ/AWGN)', 'BER_teorica_pam');
axis([min(snr_vetor) max(snr_vetor) 1e-5 1]); % Ajusta o limite inferior para ver a queda






% ============================================================
%  REFERÊNCIAS
% ============================================================
% H. Camporez et al., "AI-Driven Enhancements for Handover in Visible Light Communication Systems," in Journal of Lightwave Technology, vol. 42, no. 23, pp. 8191-8202, 1 Dec.1, 2024, doi: 10.1109/JLT.2024.3430188.
% E. d. V. Pereira, H. R. d. O. Rocha, R. B. Nunes, M. E. V. Segatto and J. A. L. Silva, "Impact of Optical Power in the Guard-Band Reduction of an Optimized DDO-OFDM System," in Journal of Lightwave Technology, vol. 33, no. 23, pp. 4717-4725, 1 Dec.1, 2015, doi: 10.1109/JLT.2015.2481085.
% PROAKIS, John G.; SALEHI, Masoud. Digital Communications. 5. ed. McGraw-Hill, 2008.
