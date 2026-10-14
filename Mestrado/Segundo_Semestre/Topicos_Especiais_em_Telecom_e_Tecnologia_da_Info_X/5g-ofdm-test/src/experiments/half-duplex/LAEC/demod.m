function [x,x2] = demod(y, Fc, Fs, varargin)
%DEMOD Signal demodulation for communications simulations.
%   X = DEMOD(Y,Fc,Fs,METHOD,OPT) demodulates the carrier signal Y with a
%   carrier frequency Fc and sampling frequency Fs, using the demodulation
%   scheme in METHOD.  OPT is an extra, sometimes optional, parameter whose
%   purpose depends on the demodulation scheme you choose.
%
%   Fs must satisfy Fs > 2*Fc + BW, where BW is the bandwidth of the
%   modulated signal.
%
%        METHOD            DEMODULATION SCHEME
%    'am',      Amplitude demodulation, double side-band, suppressed carrier
%    'amdsb-sc' OPT not used.
%    'amdsb-tc' Amplitude demodulation, double side-band,transmitted carrier
%               OPT is a scalar which is subtracted from the decoded message
%               signal.  It defaults to zero.
%    'amssb'    Amplitude demodulation, single side-band
%               OPT not used.
%    'fm'       Frequency demodulation
%               OPT is a scalar which specifies the constant of frequency
%               modulation kf, which defaults to 1.
%    'pm'       Phase demodulation
%               OPT is a scalar which specifies the constant of phase
%               modulation kp, which defaults to 1.
%    'pwm'      Pulse width demodulation
%               By setting OPT = 'centered' you tell DEMOD that the pulses
%               are centered on the carrier period rather than being
%               "left justified".
%    'ppm'      Pulse position demodulation
%               OPT is not used.
%    'qam'      Quadrature amplitude demodulation
%               For QAM signals, use [X1,X2] = DEMOD(Y,Fc,Fs,'qam')
%
%   If Y is a matrix, its columns are demodulated.
%
%   % Example:
%   %   Demodulate a received signal which is frequency modulated at
%   %   carrier frequency 3KHz.
%
%   Fs = 8000;                              % sampling frequency
%   t = (0:1000-1)/Fs;                      % time vector
%   s = 4*cos(2*pi*500*t);                  % modulating signal
%   x = modulate(s,3e3,Fs,'fm',0.1);        % modulated signal
%   rx = x + sqrt(1e-3)*randn(size(x));     % received signal
%   y = demod(rx,3e3,Fs,'fm');              % demodulated signal
%   [px,fx] = pwelch(rx,[],0,length(x),Fs); % PSD, received signal
%   [py,fy] = pwelch(y,[],0,length(y),Fs);  % PSD, demodulated signal
%
%   subplot(211);
%   plot(fx/1e3,px); grid on
%   title('Received signal')
%   ylabel('Amplitude(Watts)')
%
%   subplot(212);
%   plot(fy/1e3,py,'r'); grid on
%   title('Demodulated Signal')
%   xlabel('Frequency(KHz)');
%   ylabel('Amplitude(Watts)')
%
%   See also MODULATE PAMDEMOD QAMDEMOD GENQAMDEMOD FSKDEMOD
%            PSKDEMOD MSKDEMOD.
%
%   Note:  PAMDEMOD, QAMDEMOD, GENQAMDEMOD, FSKDEMOD, PSKDEMOD and MSKDEMOD
%          are in the Communications Toolbox.

%   Copyright 1988-2024 The MathWorks, Inc.

%#codegen

narginchk(3,5)

if nargin > 3
    method = convertStringsToChars(varargin{1});
else
    method = 'am';
end
coder.internal.assert(coder.internal.isConst(method),...
    'signal:codegeneration:OptionInputsMustBeConstant');
if nargin > 4
    P1 = convertStringsToChars(varargin{2});
end
% Checks if Y is a valid numeric data input. Cast to double since filtfilt
% does not accept single inputs.
validateattributes(y,{'single','double'},{'2d'},mfilename,'y',1);
y_in = y;


% Cast to enforce precision rules
validateattributes(Fc,{'numeric'},{'real','scalar','finite'},mfilename,'FC',2);
Fc = double(Fc);
validateattributes(Fs,{'numeric'},{'real','scalar','finite'},mfilename,'FS',3);
Fs = double(Fs);
coder.internal.errorIf(Fc>=Fs/2,'signal:demod:InvalidRange');

[r,c]=size(y_in);
if r*c == 0
    x = cast([],"like",y);
    x2 = cast([],"like",y);
    return
end


if (r==1)   % convert row vector to column
    y = y_in(:);  len = c;
else
    y = y_in;
    len = r;
end

if strcmpi(method,'am') || strcmpi(method,'amdsb-sc') || ...
        strcmpi(method,'amdsb-tc') || strcmpi(method,'amssb')
    t = (0:1/Fs:((len-1)/Fs))';
    t = repmat(t,1,size(y,2));
    x_in = y.*cos(2*pi*Fc*t);
    [b,a]=butter(5,Fc*2/Fs);
    for i = 1:size(y,2)
        x_in(:,i) = filtfilt(b,a,x_in(:,i));
    end

    if strcmpi(method,'amdsb-tc')
        if nargin < 5
            P1 = 0;
        end
        P1 = signal.internal.sigcasttofloat(P1,'double','demod','OPT',...
            'allownumeric');
        x_in = x_in - P1;
    end
    x2 = cast([],"like",y);

elseif strcmpi(method,'fm')
    if nargin < 5, P1 = 1; end
    P1 = signal.internal.sigcasttofloat(P1,'double','demod','OPT',...
        'allownumeric');
    t = (0:1/Fs:((len-1)/Fs))';
    t = repmat(t,1,size(y,2));
    yq = hilbert(y).*exp(-1i*2*pi*Fc*t);
    x_in = (1/P1)*[zeros(1,size(yq,2)); diff(unwrap(angle(yq)))];
    x2 = cast([],"like",y);

elseif strcmpi(method,'pm')
    if nargin < 5, P1 = 1; end
    P1 = signal.internal.sigcasttofloat(P1,'double','demod','OPT',...
        'allownumeric');
    t = (0:1/Fs:((len-1)/Fs))';
    t = repmat(t,1,size(y,2));
    yq = hilbert(y).*exp(-1i*2*pi*Fc*t);
    x_in = (1/P1)*angle((yq));
    x2 = cast([],"like",y);

elseif strcmpi(method,'pwm')
    % precondition input by thresholding:
    y1 = y>.5;
    t = (0:1/Fs:((len-1)/Fs))';
    len = ceil( len * Fc / Fs);   % length of message signal
    x_in = zeros(len,size(y,2),"like",y);
    if nargin < 5
        P1 = 'left';
    else
        string_method = {'left','centered'};
        validatestring(P1,string_method,mfilename,'P1',5);
    end
    isCentered=strcmpi('centered',P1);
    isLeft = strcmpi('left',P1);
    coder.internal.assert(isCentered||isLeft,'signal:demod:SignalErr')
    if isCentered
        for i = 1:len
            t_temp = t-(i-1)/Fc;
            ind =  (t_temp >= -1/2/Fc) & (t_temp < 1/2/Fc) ;
            for j1 = 1:size(y1,2)   % for each column ...
                x_in(i,j1) = sum(y1(ind,j1))*Fc/Fs;
            end
        end
        x_in(1,:) = x_in(1,:)*2;
    elseif strcmpi('left',P1)
        for i = 1:len
            t_temp = t-(i-1)/Fc;
            ind =  (t_temp >= 0) & (t_temp < 1/Fc) ;
            for j2 = 1:size(y1,2)   % for each column ...
                x_in(i,j2) = sum(y1(ind,j2))*Fc/Fs;
            end
        end

    end
    x2 = cast([],"like",y);

elseif strcmpi(method,'ptm') || strcmpi(method,'ppm')

    % precondition input by thresholding:
    y1 = y>.5;
    t = (0:1/Fs:((len-1)/Fs))'*Fc;
    len = ceil( len * Fc / Fs);   % length of message signal
    x_in = zeros(len,size(y,2),"like",y);
    for i = 1:len
        t_temp = t-(i-1);
        ind = find( (t_temp >= 0) & (t_temp<1) );
        for j3 = 1:size(y1,2)    % for each column ...
            ind2 = y1(ind,j3)==1;
            if any(ind2)
                x_in(i,j3) = t_temp(min(ind(ind2)));
            end
        end
    end
    x2 = cast([],"like",y);
elseif strcmpi(method,'qam')
    t = (0:1/Fs:((len-1)/Fs))';
    t = repmat(t,1,size(y,2));
    x_in = 2*y.*cos(2*pi*Fc*t);
    x2_in = 2*y.*sin(2*pi*Fc*t);
    [b,a]=butter(5,Fc*2/Fs);
    for i = 1:size(y,2)
        x_in(:,i) = filtfilt(b,a,x_in(:,i));
        x2_in(:,i) = filtfilt(b,a,x2_in(:,i));
    end
    if (r==1)   % convert x2 from a column to a row if necessary
        x2 = x2_in.';
    else
        x2 = x2_in;
    end

else
    coder.internal.errorIf(true,'signal:demod:InvalidMethod')
end


if (r==1)   % convert x from a column to a row
    x = x_in.';
else
    x = x_in;
end
