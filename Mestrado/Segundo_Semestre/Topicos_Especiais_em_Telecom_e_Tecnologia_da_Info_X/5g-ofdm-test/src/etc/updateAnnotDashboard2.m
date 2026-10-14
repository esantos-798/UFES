function updateAnnotDashboard2(berDL,berUL,berPayload)
%UPDATEANNOTDASHBOARD Summary of this function goes here
%   Detailed explanation goes here
pvert = 0.85;
%delete(findall(gcf,'type','annotation'))
delt_pvert = 0.04;
phor = 0.7;

% BER Downlink
pos = [phor pvert-0*delt_pvert 1 0.08];
str = ['BER Downlink no CR = ', num2str(berDL)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% BER Uplink 
pos = [phor pvert-1*delt_pvert 1 0.08];
str = ['BER Uplink no CR = ', num2str(berUL)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

% BER Payload
pos = [phor pvert-2*delt_pvert 1 0.08];
str = ['BER Downlink of payload = ', num2str(berPayload)];
t = annotation('textbox',pos,'string',str, 'EdgeColor','none');
t.FontSize = 12; t.FontName = 'Times New Roman';

end

