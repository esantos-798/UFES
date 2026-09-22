function [X,Y,Z] = polar3d(Zin,theta1,theta2,r1,r2,scale,edge)

    % Número de pontos angulares
    ntheta = size(Zin,2);

    % Número de pontos radiais
    nr = size(Zin,1);

    % Coordenadas
    theta = linspace(theta1,theta2,ntheta);
    r = linspace(r1,r2,nr);

    % Malha polar
    [Theta,R] = meshgrid(theta,r);

    % Conversão para coordenadas cartesianas
    X = R .* cos(Theta);
    Y = R .* sin(Theta);

    % Coordenada Z
    Z = Zin * scale;

end