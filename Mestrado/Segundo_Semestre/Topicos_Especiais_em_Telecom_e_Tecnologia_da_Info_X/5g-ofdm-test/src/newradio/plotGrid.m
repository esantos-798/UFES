function  plotGrid(grid)
% Plot Resource grid

% Input parameters:
% grid - Resource Grid to be plotted

% Resource Grid Plot
figure();
imagesc(abs(grid));
axis xy;
xlabel('OFDM symbol');
ylabel('Subcarrier');
%title('5G NR Resource Grid')
end

