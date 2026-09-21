clear;
clc;
close all;

% ============================================================
% Parâmetros da fibra
% ============================================================

a = 8.335;
lambda = 0.6328;
nc = 1.462420;
ng = 1.457420;

% Número V
v = (2*pi*a/lambda)*sqrt(nc^2-ng^2);

umax = v;
umin = 0;
wmax = v;
wmin = 0;

% ============================================================
% Curvas para determinação dos modos
% ============================================================

u = 0:0.1:v;
Su = length(u);

J = zeros(size(u));

for m = 1:Su
    J(m) = u(m)*besselj(1,u(m))/besselj(0,u(m));
end

w = zeros(size(u));
K = zeros(size(u));

for m = 1:Su
    w(m) = sqrt(v^2-u(m)^2);
    K(m) = w(m)*besselk(1,w(m))/besselk(0,w(m));
end

% Para visualizar as curvas, retire o comentário:
% figure;
% plot(u,J,'b','LineWidth',1.5);
% hold on;
% plot(w,K,'r','LineWidth',1.5);
% grid on;
% xlabel('u / w');
% ylabel('Função');
% legend('u J_1(u)/J_0(u)','w K_1(w)/K_0(w)');

% ============================================================
% Modos
% ============================================================

u01 = 2.1845;
neff01 = sqrt(nc^2-(u01/(2*pi*a/lambda))^2);

u02 = 4.9966;
neff02 = sqrt(nc^2-(u02/(2*pi*a/lambda))^2);

u03 = 7.7642;
neff03 = sqrt(nc^2-(u03/(2*pi*a/lambda))^2);

w01 = sqrt(v^2-u01^2);
w02 = sqrt(v^2-u02^2);
w03 = sqrt(v^2-u03^2);

% Mostrar resultados
fprintf('Número V = %.4f\n',v);

fprintf('\nModo 01:\n');
fprintf('u01 = %.4f\n',u01);
fprintf('w01 = %.4f\n',w01);
fprintf('neff01 = %.6f\n',neff01);

fprintf('\nModo 02:\n');
fprintf('u02 = %.4f\n',u02);
fprintf('w02 = %.4f\n',w02);
fprintf('neff02 = %.6f\n',neff02);

fprintf('\nModo 03:\n');
fprintf('u03 = %.4f\n',u03);
fprintf('w03 = %.4f\n',w03);
fprintf('neff03 = %.6f\n',neff03);

% ============================================================
% Campo no núcleo
% ============================================================

rc = 0:0.1:a;
Src = length(rc);

psic02 = zeros(size(rc));

for m = 1:Src
    psic02(m) = besselj(0,u02*rc(m)/a)/besselj(0,u02);
end

% ============================================================
% Campo na região da casca
% ============================================================

rg = a:0.1:2*a;
Srg = length(rg);

psig02 = zeros(size(rg));

for m = 1:Srg
    psig02(m) = besselk(0,w02*rg(m)/a)/besselk(0,w02);
end

% ============================================================
% Preparação das matrizes
% ============================================================

psic02t = psic02';
psig02t = psig02';

psic02t = psic02t(end:-1:1);
psig02t = psig02t(end:-1:1);

PSIC02 = repmat(psic02t,1,500);
PSIG02 = repmat(psig02t,1,500);

% ============================================================
% Geração das superfícies polares
% ============================================================

[Xc,Yc,Zc] = polar3d( ...
    PSIC02.^2/min(min(PSIC02))^2, ...
    0,2*pi,0,a,2,'off');

[Xg,Yg,Zg] = polar3d( ...
    PSIG02.^2/min(min(PSIC02))^2, ...
    0,2*pi,a,2*a,2,'off');

% ============================================================
% Gráficos
% ============================================================

%figure;
fig = figure('visible', 'off');
graphics_toolkit(fig, 'gnuplot');

subplot(2,1,1);

surf(Xc,Yc,Zc);
hold on;

surf(Xg,Yg,Zg);

axis([-20 20 -20 20]);
shading interp;
colorbar;

xlabel('x');
ylabel('y');
zlabel('Intensidade');
title('Distribuição espacial do modo LP_{02}');


% ============================================================

subplot(2,1,2);

pcolor(Xc,Yc,Zc);
hold on;

pcolor(Xg,Yg,Zg);

axis([-10 10 -10 10]);
shading interp;
colorbar;

xlabel('x');
ylabel('y');
title('Distribuição transversal do modo LP_{02}');

print("LP02.png","-dpng","-r300");