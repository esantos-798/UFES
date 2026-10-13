%function [h_mc, t_bins] = simular_mc_uwoc_HG()
function [h_mc, t_bins] = simular_mc_uwoc_HG_1(L)

    % --- Parâmetros de entrada -------------------------------------------
    num_fotons = 1e4;           % Reduzido para 1e4 para testes rápidos. Aumente para 1e5 depois!
    L = 20;                     % distância linear entre transmissor e receptor (m)
    a = 0.0508;                 % coeficiente de absorção (tabelado conforme o tipo de água)
    b = 0.2116;                 % coeficiente de espalhamento (tabelado conforme o tipo de água)
    v = 2.25e8;                 % velocidade da luz na água
    g = 0.85;                   % Fator de assimetria de Henyey-Greenstein
    
    c = a + b;                  % coeficiente de extinção
                              
    t_chegada = [];
    
    for i = 1:num_fotons
        pos = [0, 0, 0];
        dir = [1, 0, 0]; % Vetor direção original
        dist_total = 0;
        
        while pos(1) < L && dist_total < 5*L
            % Passo livre
            dist_passo = -log(rand()) / c;
            pos = pos + dir * dist_passo;
            dist_total = dist_total + dist_passo;
            
            % Dispersão via Henyey-Greenstein
            r = rand();
            cos_theta = (1/(2*g)) * (1 + g^2 - ((1-g^2)/(1-g+2*g*r))^2);
            sin_theta = sqrt(1 - cos_theta^2);
            phi = 2*pi*rand();

            % Atualiza vetor direção (matriz de rotação)
            nova_dir = [cos_theta, sin_theta*cos(phi), sin_theta*sin(phi)];
            dir = nova_dir / norm(nova_dir);
        end
        
        if pos(1) >= L % O fóton atingiu o plano do detector
            t_chegada = [t_chegada, dist_total / v];
        end
    end
    
    % --- Substituição robusta e compatível para o histcounts ---
    num_bins = 50;
    if isempty(t_chegada)
        error('Nenhum fóton chegou ao detector. Tente aumentar o num_fotons ou diminuir a distância L.');
    end
    
    % Calcula o histograma usando os centros dos bins (compatível com qualquer Octave/MATLAB)
    [contagens, centros_bins] = hist(t_chegada, num_bins);
    
    % Normalização manual para PDF (Área sob a curva = 1)
    largura_bin = centros_bins(2) - centros_bins(1);
    h_mc = contagens / (length(t_chegada) * largura_bin);
    
    % Reconstrói as bordas dos intervalos (t_bins) para manter compatibilidade com o script principal
    t_bins = [centros_bins - largura_bin/2, centros_bins(end) + largura_bin/2];
    
    % --- Plotagem do Monte Carlo ---
    figure('Color', 'w');
    bar(centros_bins, h_mc, 'BarWidth', 1);
    title('Resposta ao Impulso via Monte Carlo');
    xlabel('Tempo (s)'); ylabel('PDF');
    grid on;
end