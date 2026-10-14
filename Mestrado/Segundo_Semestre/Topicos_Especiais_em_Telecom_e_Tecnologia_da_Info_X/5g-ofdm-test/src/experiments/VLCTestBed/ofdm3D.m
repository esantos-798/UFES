% Parâmetros OFDM
num_subportadoras = 64; % Número de subportadoras
espacamento_subportadoras = 15e3; % Espaçamento entre subportadoras
frequencia_simbolica = 64*15e3; % Frequência de símbolo

% Parâmetros da modulação QAM
M = 16; % Número de pontos de constelação MQAM (16-QAM)

% Sequência de dados aleatória para QAM
dados_qam = randi([0, M-1], 1, num_subportadoras); % Sequência de símbolos QAM

% Mapeia os símbolos QAM para a constelação
constelacao_qam = qammod(dados_qam, M);

% Zero padding para corresponder ao número de subportadoras
sinal_ofdm = zeros(1, num_subportadoras);

% Insira os símbolos modulados QAM nas subportadoras
sinal_ofdm(1:num_subportadoras) = constelacao_qam;

% Realize a IFFT para obter o sinal no domínio do tempo
sinal_temporal = ifft(sinal_ofdm);

% Calcula as frequências das subportadoras
frequencias_subportadoras = (0:num_subportadoras-1) * espacamento_subportadoras;

% Cria matrizes de índices de subportadoras
indice_subportadoras = repmat(frequencias_subportadoras', 1, num_subportadoras);

% Inicializa uma matriz de amplitudes
amplitudes_subportadoras = zeros(num_subportadoras, num_subportadoras);

% Calcula as amplitudes para cada combinação de índice de subportadora e símbolo QAM
for i = 1:num_subportadoras
    for j = 1:num_subportadoras
        amplitudes_subportadoras(j, i) = abs(sin(2 * pi * frequencias_subportadoras(j) / frequencia_simbolica)) * abs(constelacao_qam(i));
    end
end

% Duração de cada símbolo em segundos
duracao_simbolo = 1 / frequencia_simbolica;

% Cria matrizes de índices de tempo (duração de cada símbolo)
tempo_simbolos = 0:num_subportadoras-1;

% Plota a figura característica do OFDM em 3D
figure;
surf(tempo_simbolos * duracao_simbolo, frequencias_subportadoras, amplitudes_subportadoras);
xlabel('Duração de Cada Símbolo (s)');
ylabel('Domínio da Frequência');
zlabel('Amplitude');
title('Figura Característica do OFDM com Modulação QAM em 3D');
colorbar;

print("test.png","-dpng","-r300");

% Personaliza a aparência da figura
colormap('jet');
shading('interp');

% Ajusta os eixos
axis([0, max(tempo_simbolos) * duracao_simbolo, min(frequencias_subportadoras), max(frequencias_subportadoras), 0, max(amplitudes_subportadoras(:))]);