% Parâmetros OFDM
numSubcarriers = 64;         % Número de subportadoras
prefixLength = 0;           % Tamanho do prefixo cíclico

% Gerar símbolos OFDM aleatórios
numSymbols = 64;            % Número de símbolos OFDM
ofdmSymbols = randi([0, 1], numSubcarriers, numSymbols);

% Transformada Inversa de Fourier (IFFT)
timeDomainSymbols = ifft(ofdmSymbols, numSubcarriers);

% Adicionar prefixo cíclico
ofdmTxSignal = [timeDomainSymbols(end - prefixLength + 1:end, :); timeDomainSymbols];

% Criar grades de frequência e tempo
frequencies = 0:(numSubcarriers - 1);
timeIndices = 0:(numSubcarriers + prefixLength- 1);

% Criar uma malha 3D
[timeGrid, freqGrid] = meshgrid(timeIndices, frequencies);

% Plot 3D
%figure;
fig1 = figure('visible', 'off');
graphics_toolkit(fig1, 'gnuplot');
mesh(freqGrid, timeGrid, abs(ofdmTxSignal));
title('Símbolos OFDM no Domínio da Frequência e do Tempo');
xlabel('Subportadora');
ylabel('Tempo');
zlabel('Magnitude');

print("test.png","-dpng","-r300");

% Ajustar os eixos para melhor visualização
axis tight;

% Mostrar a legenda da cor para o valor absoluto
colorbar;

% Rotação da visualização para melhorar a visualização
view(40, 30);