% Salve este arquivo exatamente como: scatterplot.m
function h = custom_scatterplot(x, varargin)
    % SCATTERPLOT Versão customizada e altamente compatível com Octave Puro
    
    % Se o sinal vier em formato de matriz, transforma em vetor coluna
    x = x(:);
    
    % Extrai a parte real (In-Phase) e imaginária (Quadrature)
    I = real(x);
    Q = imag(x);
    
    % Abre uma nova figura de gráfico
    figure;
    
    % Plota os pontos na tela como pontos azuis
    plot(I, Q, 'b.', 'MarkerSize', 8); 
    
    % Configurações estéticas usando a função 'set' (À prova de falhas no Octave)
    grid on;
    ax = gca();
    set(ax, 'XAxisLocation', 'origin'); % Eixo X cruzando no zero
    set(ax, 'YAxisLocation', 'origin'); % Eixo Y cruzando no zero
    
    % Dá um pequeno respiro nas bordas do gráfico
    max_val = max(max(abs(I)), max(abs(Q))) * 1.2;
    if max_val > 0
        xlim([-max_val, max_val]);
        ylim([-max_val, max_val]);
    end
    
    xlabel('Em Fase (In-Phase I)');
    ylabel('Quadratura (Quadrature Q)');
    title('Constelação do Sinal Recebido');
    
    % Retorna o handle da figura caso o script precise
    if nargout > 0
        h = gcf;
    end
end