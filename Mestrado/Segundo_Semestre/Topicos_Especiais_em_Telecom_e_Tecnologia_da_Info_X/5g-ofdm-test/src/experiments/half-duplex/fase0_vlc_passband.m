function varargout = fase0_vlc_passband(action, varargin)
%FASE0_VLC_PASSBAND DUC/DDC didático para o ramo VLC/b2b do main_downlink.
%
%   O main_downlink usa modulate(...,'qam',...) e demod(...) quando existem
%   (tipicamente MATLAB com funções do projeto). Aqui há um fallback explícito
%   para você ver a matemática da portadora em 5 MHz.
%
%   y = fase0_vlc_passband('duc', x_bb, fc, Fs)
%   x_bb = fase0_vlc_passband('ddc', y_pb, fc, Fs)
%
%   x_bb  — sinal complexo em banda base (após upsample)
%   y_pb  — sinal REAL passa-faixa (como entraria no DAC / LED)
%   fc    — pVLCCenterFrequency (ex.: 5e6 Hz)
%   Fs    — pDACSampleRate (ex.: 75e6 Hz)

    switch lower(action)
        case 'duc'
            x_bb = varargin{1};
            fc = varargin{2};
            Fs = varargin{3};
            x_bb = x_bb(:);
            n = (0:numel(x_bb)-1)';
            t = n / Fs;
            % Mesma ideia do modulate QAM analógico: parte real de (I+jQ) e^(j 2 pi fc t)
            y_pb = real(x_bb) .* cos(2*pi*fc*t) - imag(x_bb) .* sin(2*pi*fc*t);
            varargout{1} = y_pb;

        case 'ddc'
            y_pb = varargin{1};
            fc = varargin{2};
            Fs = varargin{3};
            y_pb = y_pb(:);
            n = (0:numel(y_pb)-1)';
            t = n / Fs;
            % Mistura down + combinação quadratura (sem filtro passa-baixa ideal)
            xr = y_pb .* cos(2*pi*fc*t);
            xi = -y_pb .* sin(2*pi*fc*t);
            x_bb = xr + 1i*xi;
            varargout{1} = x_bb;

        otherwise
            error('fase0_vlc_passband:Action', 'Use ''duc'' ou ''ddc''.');
    end
end
