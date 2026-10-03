function [sbpa, t] = gerador_sequencia_binaria_aleatoria(delta, tf) 

t=[0:delta:tf];sna=rand(size(t));l=0;
while l==0
    alfa=0.5;
    if alfa<=1
        if alfa>=0
            l=1;
        end
    end
    if l==0
        disp('Alfa deve estar entre 0 e 1');
    end
end
for i=1:length(t),
    if sna(i)<=alfa,sbpa(i)=-1;
    else sbpa(i)=1;
    end
end
fig = figure('visible', 'off');
graphics_toolkit(fig, 'gnuplot');
plot(t,sbpa);
print(fig, 'sequencia_binaria_aleatoria.png', '-dpng');