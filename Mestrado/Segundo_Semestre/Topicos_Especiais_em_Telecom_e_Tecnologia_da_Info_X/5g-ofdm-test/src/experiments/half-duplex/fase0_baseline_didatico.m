function fase0_baseline_didatico()
%FASE0_BASELINE_DIDATICO Baseline didático do main_downlink (Fase 0, antes do UWOC).
%
%   fase0_baseline_didatico
%
%   Roda em MATLAB (com ou sem 5G Toolbox) e no GNU Octave.
%   Arquivo em formato FUNCTION (subfunções no fim) — Octave não aceita
%   subfunções em scripts .m comuns.
%
%   Edite as variáveis FASE0_* no corpo desta função antes de executar.

% =========================================================================
% FASE 0 — Baseline didático do main_downlink (antes de inserir UWOC)
% =========================================================================

clear; close all; clc;

%% --- Configuração da Fase 0 (edite aqui) ---
FASE0_INTERATIVO    = false;   % pausa com Enter entre blocos explicativos
FASE0_GERAR_FIGURAS = true;   % gráficos de espectro (16-QAM)
FASE0_SEED          = 42;     % mesma semente em todos os testes da tabela

modos = {
    'QPSK',   4;
    '16QAM', 16;
    '64QAM', 64;
};
canais = {'software', 'b2b'};

%% PARTE 1
fase0_imprimir_cabecalho('PARTE 1: O que o main_downlink.m faz?');

fprintf([ ...
    'Pense em três blocos encadeados:\n\n' ...
    '  [A] TRANSMissor 5G NR\n' ...
    '      Bits → codificação LDPC → símbolos QAM → grade de recursos\n' ...
    '      (SSB para sincronizar + PDCCH controle + PDSCH dados)\n' ...
    '      → nrOFDMModulate → waveform (complexo, banda base, ~7,68 MS/s)\n\n' ...
    '  [B] CANAL (pChannel)\n' ...
    '      software : permanece em banda base complexa + ruído + erro de frequência\n' ...
    '      b2b      : sobe taxa (75 MS/s), modula portadora 5 MHz (sinal REAL),\n' ...
    '                 loopback perfeito, desce de volta à banda base\n' ...
    '      adalm2000: igual b2b, mas DAC/ADC reais (luz branca / VLC terrestre)\n\n' ...
    '  [C] RECEPTor 5G NR\n' ...
    '      Achar início do frame (SSB) → corrigir CFO → OFDM demod → equalizar\n' ...
    '      → decodificar bits → BER e EVM + gráficos que você já usa\n\n' ...
    'A Fase 0 só compara [B] software vs [B] b2b. O bloco [A] e [C] ficam iguais.\n' ...
    'Depois, na adaptação UWOC, vamos trocar o miolo de [B] por LED + canal subaquático.\n\n']);

fase0_pausar(FASE0_INTERATIVO);

%% PARTE 2
fase0_imprimir_cabecalho('PARTE 2: Parâmetros fixos nesta Fase 0');

fprintf([ ...
    '  pSCS = 15 kHz        → numerologia mu=0 (símbolo OFDM mais longo, estilo LTE)\n' ...
    '  pSizeGrid = 25 RB    → 25 x 12 = 300 subportadoras úteis\n' ...
    '  Largura ≈ 15 kHz x 12 x 25 = 4,5 MHz (grade); taxa amostral NR ≈ 7,68 MS/s\n\n' ...
    '  pSNRdB = 50 (software) → canal quase limpo; BER deve ir a ~0 se LDPC convergir\n\n' ...
    '  pVLCCenterFrequency = 5 MHz, pDACSampleRate = 75 MS/s (só no b2b)\n' ...
    '      → emula o caminho VLC da dissertação ANTES de hardware M2K\n\n']);

fase0_pausar(FASE0_INTERATIVO);

%% PARTE 3
fase0_imprimir_cabecalho('PARTE 3: Medindo BER / EVM (software vs b2b)');

R0 = fase0_downlink_metricas(struct('pChannel', 'software', 'pDigitalModulation', '16QAM', ...
    'pModulationLevel', 16, 'seed', FASE0_SEED, 'verbose', false));
if R0.has5GToolbox
    fprintf('Ambiente: MATLAB com 5G Toolbox (BER pós-LDPC).\n\n');
else
    fprintf('Ambiente: custom_* / Octave (BER pré-FEC — compare tendências, não valor absoluto com a tese).\n\n');
end

resultados = cell(size(modos, 1)*numel(canais), 1);   % numel(modos) = 6 (matriz 3x2), nao 3
idx = 0;
for im = 1:size(modos, 1)
    for ic = 1:numel(canais)
        idx = idx + 1;
        cfg = struct( ...
            'pChannel', canais{ic}, ...
            'pDigitalModulation', modos{im, 1}, ...
            'pModulationLevel', modos{im, 2}, ...
            'seed', FASE0_SEED, ...
            'verbose', true);
        resultados{idx} = fase0_downlink_metricas(cfg);
    end
end

fase0_imprimir_tabela(resultados);

fprintf([ ...
    '\nComo ler a tabela:\n' ...
    '  • software + SNR alto: BER ~ 0 e EVM baixo → receptor NR está saudável.\n' ...
    '  • b2b: BER similar ao software → DUC/DDC não quebram o enlace (passo VLC OK).\n' ...
    '  • Se b2b piorar muito: investigar resample/portadora antes de UWOC.\n' ...
    '  • CFO_total (correção aplicada) no software deve ser ~ -1200 Hz (sinal oposto ao CFO de +1200 Hz injetado); no b2b ~ 0.\n\n']);

T = fase0_structs_to_table(resultados);
fase0_assign_base('FASE0_TABELA', T);
fase0_assign_base('FASE0_RESULTADOS', resultados);
fprintf('Tabela salva no workspace como FASE0_TABELA.\n\n');

fase0_pausar(FASE0_INTERATIVO);

%% PARTE 4
fase0_imprimir_cabecalho('PARTE 4: Relação com os gráficos do main_downlink.m');

fprintf([ ...
    'Quando você roda main_downlink.m, as figuras correspondem a:\n\n' ...
    '  Espectro TX (subplot 1) → waveform após nrOFDMModulate (banda base)\n' ...
    '  Espectro RX (subplot 2) → rxWaveform após o canal, antes do CFO fino\n' ...
    '  Grade de recursos      → DLgrid (onde SSB, PDCCH e PDSCH ocupam REs)\n' ...
    '  Constelações 2x2     → TX; RX só timing; RX timing+freq; RX equalizado\n' ...
    '  mesh(hest)             → canal estimado por DMRS (no software ≈ flat)\n\n' ...
    'No modo adalm2000/b2b há figuras extras em 75 MS/s (passa-faixa em ~5 MHz).\n' ...
    'Essas são as mesmas etapas que o UWOC usará depois (com LED + dispersão no meio).\n\n']);

fase0_pausar(FASE0_INTERATIVO);

%% PARTE 5
if FASE0_GERAR_FIGURAS
    fase0_imprimir_cabecalho('PARTE 5: Figuras comparativas (16-QAM)');

    Rs = fase0_downlink_metricas(struct('pChannel', 'software', 'pDigitalModulation', '16QAM', ...
        'pModulationLevel', 16, 'seed', FASE0_SEED));
    Rb = fase0_downlink_metricas(struct('pChannel', 'b2b', 'pDigitalModulation', '16QAM', ...
        'pModulationLevel', 16, 'seed', FASE0_SEED));

    fase0_plot_espectros(Rs, Rb);
    fprintf('Figura 1: espectros TX/RX — software vs b2b (16-QAM, seed=%d).\n', FASE0_SEED);
    fprintf('Compare: b2b deve mostrar energia concentrada em torno de ±5 MHz no passa-faixa.\n\n');
end

fase0_imprimir_cabecalho('Fase 0 concluída');
fprintf([ ...
    'Próximo passo (Fase 1): extrair LED + Double-Gamma em funções reutilizáveis\n' ...
    'e criar pChannel ''uwoc-b2b'' entre DUC e DDC.\n\n' ...
    'Exercício manual recomendado:\n' ...
    '  1) Abra main_downlink.m e rode com pChannel=''software'' e depois ''b2b''.\n' ...
    '  2) Confira se BER/EVM batem com FASE0_TABELA (mesma modulação).\n' ...
    '  3) Anote no caderno o que mudou nos espectros e nas constelações.\n\n']);

end

%% ------------------------------------------------------------------------
function fase0_imprimir_cabecalho(titulo)
    fprintf('\n');
    fprintf('================================================================\n');
    fprintf('  %s\n', titulo);
    fprintf('================================================================\n\n');
end

function fase0_pausar(ativo)
    if ativo
        input('  [Enter para continuar]', 's');
        fprintf('\n');
    end
end

function fase0_imprimir_tabela(resultados)
    disp(' ');
    disp('========== TABELA FASE 0 — guarde estes números como baseline ==========');
    fprintf('%-8s %-10s %12s %10s %8s %12s\n', ...
        'Mod', 'Canal', 'BER', 'EVM_%', 'Timing', 'CFO_Hz');
    fprintf('%s\n', repmat('-', 1, 72));
    for k = 1:numel(resultados)
        r = resultados{k};
        fprintf('%-8s %-10s %12.4g %10.2f %8d %12d\n', ...
            r.pDigitalModulation, r.pChannel, r.ber, r.evm_pct, ...
            r.timingOffset, r.cfo_coarse_hz + r.cfo_fine_hz);
    end
    disp('========================================================================');
end

function T = fase0_structs_to_table(resultados)
    n = numel(resultados);
    modu = cell(n, 1);
    canal = cell(n, 1);
    ber = zeros(n, 1);
    evm = zeros(n, 1);
    tim = zeros(n, 1);
    cfo = zeros(n, 1);
    for k = 1:n
        r = resultados{k};
        modu{k} = r.pDigitalModulation;
        canal{k} = r.pChannel;
        ber(k) = r.ber;
        evm(k) = r.evm_pct;
        tim(k) = r.timingOffset;
        cfo(k) = r.cfo_coarse_hz + r.cfo_fine_hz;
    end
    if exist('table', 'builtin') || exist('table', 'file')
        try
            T = table(modu, canal, ber, evm, tim, cfo, ...
                'VariableNames', {'Modulacao', 'Canal', 'BER', 'EVM_pct', 'Timing_amostras', 'CFO_total_Hz'});
            return;
        catch
        end
    end
    T = struct('Modulacao', {modu{:}}, 'Canal', {canal{:}}, 'BER', ber, ...
        'EVM_pct', evm, 'Timing_amostras', tim, 'CFO_total_Hz', cfo);
end

function fase0_assign_base(name, value)
    try
        assignin('base', name, value);
    catch
        assignin('caller', name, value);
    end
end

function fase0_plot_espectros(Rs, Rb)
    if exist('FFT_pot2', 'file') == 0
        warning('fase0:NoFFT', 'FFT_pot2 não encontrado; figuras de espectro omitidas.');
        return;
    end

    figure('Name', 'Fase 0 — Espectros didáticos', 'Position', [80 80 1100 700]);

    subplot(2, 2, 1);
    [S, ~, f, ~] = FFT_pot2(Rs.waveform.', 1/Rs.sampleRate_nr);
    plot(f/1e6, fftshift(fase0_mag2db(abs(S)))); grid on;
    title('TX banda base — canal software');
    xlabel('Frequência (MHz)'); ylabel('PSD (dB/Hz)');

    subplot(2, 2, 2);
    [S, ~, f, ~] = FFT_pot2(Rs.rxWaveform.', 1/Rs.sampleRate_nr);
    plot(f/1e6, fftshift(fase0_mag2db(abs(S)))); grid on;
    title(sprintf('RX banda base — software (EVM %.1f%%)', Rs.evm_pct));
    xlabel('Frequência (MHz)'); ylabel('PSD (dB/Hz)');

    subplot(2, 2, 3);
    [S, ~, f, ~] = FFT_pot2(Rs.waveform.', 1/Rs.sampleRate_nr);
    plot(f/1e6, fftshift(fase0_mag2db(abs(S)))); grid on;
    title('TX banda base — mesma forma de onda antes do DUC');
    xlabel('Frequência (MHz)'); ylabel('PSD (dB/Hz)');

    if isfield(Rb, 'waveform_pb') && ~isempty(Rb.waveform_pb)
        subplot(2, 2, 4);
        Fs_dac = 75e6;
        [S, ~, f, ~] = FFT_pot2(Rb.waveform_pb(:).', 1/Fs_dac);
        plot(f/1e6, fftshift(fase0_mag2db(abs(S)))); grid on;
        title(sprintf('Passa-faixa b2b (portadora 5 MHz, EVM %.1f%%)', Rb.evm_pct));
        xlabel('Frequência (MHz)'); ylabel('PSD (dB/Hz)');
    else
        subplot(2, 2, 4);
        [S, ~, f, ~] = FFT_pot2(Rb.rxWaveform.', 1/Rb.sampleRate_nr);
        plot(f/1e6, fftshift(fase0_mag2db(abs(S)))); grid on;
        title('RX banda base — b2b');
        xlabel('Frequência (MHz)'); ylabel('PSD (dB/Hz)');
    end
end

function y = fase0_mag2db(x)
    if exist('mag2db', 'file') == 2
        y = mag2db(x);
    else
        y = 20*log10(max(x, eps));
    end
end
