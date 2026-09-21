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

% ── Parâmetros do modelo ─────────────────────────────────────────────────
Lum_max = 755;          % Iluminância máxima  [lux]
zeta  = 20;             % Parâmetro de saturação  (adimensional, > 0)
k     = 1.9;            % Fator de forma (inteiro positivo)
SNR_dB = 9.6;            % Relação sinal-ruído
R = 0.6;                % Responsatividade (Camporez, et al. 2024)
N_bits  = 3000;         % Número de bits da sequência
I_bias = 800;           % Amplitude da coorente da portadora
I_mod_A = 50;           % Amplitude da corrente modulante [mA]
%                           bit=1 → +I_mod_A  |  bit=0 → -I_mod_A  (NRZ bipolar)
alpha = 10;
Tempo_pulso = 4;        % Amostras por bit (inteiro)

% ── Parâmetros Resposta ao impulso ───────────────────────────────────────
T_ns = 16e-9;             % Janela temporal [s]

% Coeficientes Double Gamma
C1 = 4.777160;
C2 = 3.335616;
C3 = 0.529837;
C4 = 0.799894;

% ── Parâmetros da taxa de transmissão ────────────────────────────────────
Rb = 1000e6;               % Taxa de transmissão (bits/s)

Rs = Rb;                % Considerando 1 amostra por bit
Ts = 1/Rs;              % Taxa de amostragem

% ============================================================
%  CURVA DE LUMINÂNCIA
% ============================================================

% ─── Corrente de entrada ────────────────────────────────────
I_LED = linspace(0, 2500, 500);   % Varredura [mA]

% ─── Função do modelo ─ Cálculod da luminância ──────────────
Lum = I_LED ./ (zeta + (I_LED / Lum_max).^(2*k)).^(1/(2*k));


% ============================================================
%  MODULAÇÃO DO LED  —  I_fin = I_bias + I_mod 
% ============================================================

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


% ==========================================================
% Vetor da resposta ao impulso - Double Gamma
% ==========================================================

% Vetor de tempo em nanossegundos
t_h = 0:Ts:T_ns;
t_h_ns = t_h*1e9;

% Resposta ao impulso - Double Gamma
h = C1 .* t_h_ns .* exp(-C2 .* t_h_ns) + C3 .* t_h_ns .* exp(-C4 .* t_h_ns);
h = h / sum(h);    %normalização de h

% Superamostragem de h
% h_s = rectpulse(h, Tempo_pulso);


% 5) Convolução entre o sinal h e Lum_fin
y = conv(Lum_fin, h,"same");
%y = filter(h,1,Lum_fin);
%y = filtfilt(h,1,Lum_fin);
%y = Lum_fin;
% ==========================================================

% 6) Modelagem de fotodetecção
Ir = R*abs(y).^2;                     % (Pereira, et al.2015)    
                                            % Ir -> Corrente receptor
% 7) DC Block
Irb = Ir - mean(Ir);                        %retira o valor médio
 
% 8) Aplicando ruído na corrente recebida
SNR_dB = SNR_dB + 10*log10(Tempo_pulso);
Ir_ruido = awgn(Irb, SNR_dB, 'measured');

% 9) Retirando larguras de Ir_ruido
Ir_ruido_down = intdump(Ir_ruido,Tempo_pulso);
    
% 10) Verifica se o valor é positivo ou negativo
% if Ir_ruido_down > 0
%     bits_finais = 1;
% else
%     bits_finais = 0;
% end
bits_finais = Ir_ruido_down > 0;


% 11) Verifica quantos bits estão errados
erros = 0;
for i = 1:length(bits_finais)
    if bits(i) ~= bits_finais(i)
        erros = erros + 1;
    end
end

BER = erros/length(bits_finais);



% =========================================================
%  GRÁFICOS e PRINTS
% =========================================================
fprintf('Numeros de amostras da CIR = %d\n', length(t_h));


% ---------------------------------------------------------
% BER
% ---------------------------------------------------------
fprintf('BER = %.9f\n', BER);

% ---------------------------------------------------------
% Resposta ao impulso
% ---------------------------------------------------------
plot(t_h, h, 'LineWidth', 2);
grid on;

xlabel('Tempo (ns)');
ylabel('h(t)');
title('Resposta ao Impulso - Double Gamma');

% 8) Diagrama de olho
figure()
eyediagram(Ir_ruido, 2*Tempo_pulso); 
title(['Diagrama do Olho com Ruído (SNR: ', num2str(SNR_dB), 'dB)']);


% ============================================================
%  REFERÊNCIAS
% ============================================================
% H. Camporez et al., "AI-Driven Enhancements for Handover in Visible Light Communication Systems," in Journal of Lightwave Technology, vol. 42, no. 23, pp. 8191-8202, 1 Dec.1, 2024, doi: 10.1109/JLT.2024.3430188.
% E. d. V. Pereira, H. R. d. O. Rocha, R. B. Nunes, M. E. V. Segatto and J. A. L. Silva, "Impact of Optical Power in the Guard-Band Reduction of an Optimized DDO-OFDM System," in Journal of Lightwave Technology, vol. 33, no. 23, pp. 4717-4725, 1 Dec.1, 2015, doi: 10.1109/JLT.2015.2481085.
% PROAKIS, John G.; SALEHI, Masoud. Digital Communications. 5. ed. McGraw-Hill, 2008.
