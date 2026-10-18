function fase1_validacao()
%FASE1_VALIDACAO Validacao da Fase 1 (UWOC) com o receptor NR.
%
%   fase1_validacao
%
%   Roda o downlink com pChannel = 'uwoc-b2b' (uwoc_vlc_chain) e varre:
%     A) SNR eletrico   B) excursao de corrente do LED (nao linearidade/clipping)
%     C) lei do fotodetector (linear vs square)   D) CIR 'sample' vs 'integrate'
%   Funciona em MATLAB (com ou sem 5G Toolbox) e Octave. Mesma semente em todos os casos.
%   Sem 5G Toolbox o BER e pre-FEC (compare tendencias).

    SEED = 42;
    base = struct('seed', SEED, 'verbose', false);

    fprintf('\n=== Referencia: b2b ideal (Fase 0) ===\n');
    for mod = {'QPSK', 4; '16QAM', 16; '64QAM', 64}'
        r = run_case(base, 'b2b', mod{1}, mod{2}, struct());
        fprintf('  %-6s b2b ideal            BER=%-9.4g EVM=%5.2f%%\n', mod{1}, r.ber, r.evm_pct);
    end

    fprintf('\n=== A) Varredura de SNR eletrico (LED default, fotodetector linear, CIR DGF) ===\n');
    snrs = [Inf 30 20 15 10 5 0];
    fprintf('  %-6s', 'Mod'); fprintf(' %12s', arrayfun(@(s) sprintf('SNR=%g', s), snrs, 'UniformOutput', false){:}); fprintf('   (BER / EVM%%)\n');
    for mod = {'QPSK', 4; '16QAM', 16; '64QAM', 64}'
        fprintf('  %-6s', mod{1});
        for s = snrs
            r = run_case(base, 'uwoc-b2b', mod{1}, mod{2}, struct('SNR_dB', s));
            fprintf(' %5.3g/%5.1f', r.ber, r.evm_pct);
        end
        fprintf('\n');
    end

    fprintf('\n=== B) Excursao do LED (16QAM, sem ruido): nao linearidade e clipping ===\n');
    fprintf('  %-9s %-11s %-9s %-10s %-9s %-8s\n', 'I_mod_A', 'swing_mA', 'clip_%', 'Imin_mA', 'BER', 'EVM_%');
    for im = [5 20 50 100 150]
        r = run_case(base, 'uwoc-b2b', '16QAM', 16, struct('SNR_dB', Inf, 'I_mod_A', im));
        u = r.uwoc;
        fprintf('  %-9g %-11g %-9.2f %-10.0f %-9.4g %-8.2f\n', im, u.params.alpha*im, 100*u.clipFrac, u.I_min, r.ber, r.evm_pct);
    end

    fprintf('\n=== C) Lei do fotodetector (16QAM, sem ruido) ===\n');
    for im = [20 50]
        for law = {'linear', 'square'}
            r = run_case(base, 'uwoc-b2b', '16QAM', 16, struct('SNR_dB', Inf, 'I_mod_A', im, 'pdLaw', law{1}));
            fprintf('  I_mod_A=%-3g  %-7s BER=%-9.4g EVM=%5.2f%%\n', im, law{1}, r.ber, r.evm_pct);
        end
    end

    fprintf('\n=== D) Amostragem da CIR a 75 MS/s (16QAM, sem ruido) ===\n');
    for m = {'integrate', 'sample'}
        r = run_case(base, 'uwoc-b2b', '16QAM', 16, struct('SNR_dB', Inf, 'cirMethod', m{1}));
        fprintf('  %-9s h=%s  timing=%d  BER=%.4g  EVM=%5.2f%%\n', m{1}, mat2str(r.uwoc.h, 3), r.timingOffset, r.ber, r.evm_pct);
    end
    fprintf('\nNotas: SNR eletrico = potencia AC / ruido branco em 0..Fs/2 (37,5 MHz); a banda util e ~4,5 MHz.\n\n');
end

function r = run_case(base, canal, modName, modLevel, uw)
    cfg = base;
    cfg.pChannel = canal;
    cfg.pDigitalModulation = modName;
    cfg.pModulationLevel = modLevel;
    cfg.uwoc = uw;
    evalc('r = fase0_downlink_metricas(cfg);');
end
