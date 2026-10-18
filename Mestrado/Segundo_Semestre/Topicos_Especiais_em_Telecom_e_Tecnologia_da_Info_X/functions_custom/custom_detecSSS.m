function [NID1, ncellid] = custom_detecSSS(sssRx, NID2, plotConfig)
    % custom_detecSSS Replica o detecSSS.m do laboratorio (mesma metrica, mesma saida, mesmo grafico).
    %
    % Para cada NID1 (0..335) correlaciona o SSS recebido com a sequencia de referencia:
    %   sssEst(NID1+1) = sum( |mean(sssRx .* conj(sssRef), 1)|.^2 )
    % NID1 = argmax; ncellid = 3*NID1 + NID2. Imprime " Cell identity: N" e, se plotConfig,
    % faz o stem de 'SSS Correlations (Frequency Domain)'.

    if any(~isfinite(sssRx(:)))
        error('custom_detecSSS:NonFinite', 'sssRx contem NaN/Inf: nao ha como detectar o SSS.');
    end
    sssEst = zeros(1, 336);
    for n = 0:335
        sssRef = custom_nrSSS(n, NID2);
        sssEst(n+1) = sum(abs(mean(sssRx .* conj(sssRef), 1)).^2);
    end

    NID1 = find(sssEst == max(sssEst)) - 1;
    NID1 = NID1(1);
    ncellid = (3*NID1) + NID2;

    disp([' Cell identity: ' num2str(ncellid)])

    if nargin > 2 && plotConfig
        figure;
        stem(0:335, sssEst, 'o');
        title('SSS Correlations (Frequency Domain)');
        xlabel('$N_{ID}^{(1)}$', 'Interpreter', 'latex');
        ylabel('Magnitude');
        axis([-1 336 0 max(sssEst)*1.1]);
        hold on;
        plot(NID1, max(sssEst), 'kx', 'LineWidth', 2, 'MarkerSize', 8);
        try
            legend({'correlations', ['$N_{ID}^{(1)}$ = ' num2str(NID1)]}, 'Interpreter', 'latex');
        catch
            legend({'correlations', ['N_ID^(1) = ' num2str(NID1)]});
        end
    end
end
