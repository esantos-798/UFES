function print_downlink_result(R)
%PRINT_DOWNLINK_RESULT Imprime o resumo de uma execucao de run_downlink.
    fprintf('\n=== downlink 5G NR | canal %s | %s | backend %s ===\n', R.pChannel, R.pDigitalModulation, R.backend);
    fprintf('%s\n', R.channelNotes);
    fprintf('BER (%s) = %.4g  (%d erros)\n', R.berType, R.ber, R.num_errors);
    fprintf('EVM = %.2f%% | sem a subportadora de DC = %.2f%% | sem DC +-2 = %.2f%%\n', R.evm_pct, R.evm_noDC_pct, R.evm_noDCnb_pct);
    fprintf('SNR nas subportadoras uteis = %.1f dB\n', R.snr_rx_dB);
    fprintf('timing = %d amostras | CFO aplicado = %d %+d Hz\n\n', R.timingOffset, R.cfo_coarse_hz, R.cfo_fine_hz);
end
