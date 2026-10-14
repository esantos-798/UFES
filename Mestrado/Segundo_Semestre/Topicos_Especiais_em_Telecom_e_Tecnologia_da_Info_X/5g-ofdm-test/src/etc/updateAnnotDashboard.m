function updateAnnotDashboard(berDL,berUL,evmDL,evmUL,iteration,retryCount)
%UPDATEANNOTDASHBOARD Summary of this function goes here
%   Detailed explanation goes here
pvert = 0.85;
delete(findall(gcf,'type','annotation'))
delt_pvert = 0.04;
phor = 0.525;

% BER Downlink
pos = [phor pvert-0*delt_pvert 1 0.08];
str = ['BER Downlink = ', num2str(berDL)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% BER Uplink 
pos = [phor pvert-1*delt_pvert 1 0.08];
str = ['BER Uplink = ', num2str(berUL)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% EVM Downlink
pos = [phor pvert-2*delt_pvert 1 0.08];
str = ['EVM Downlink = ', num2str(evmDL), ' %'];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% EVM Uplink
pos = [phor pvert-3*delt_pvert 1 0.08];
str = ['EVM Uplink = ', num2str(evmUL), ' %'];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% Iteration Counter
pos = [phor pvert-4*delt_pvert 1 0.08];
str = ['Iteration = ', num2str(iteration)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% Retransmission Counter
pos = [phor pvert-5*delt_pvert 1 0.08];
str = ['Retransmissions = ', num2str(retryCount)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';
end

