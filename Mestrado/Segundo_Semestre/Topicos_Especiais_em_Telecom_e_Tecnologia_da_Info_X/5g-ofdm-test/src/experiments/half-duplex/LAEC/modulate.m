function [y,t] = modulate(x,Fc,Fs,varargin)
%MODULATE  Signal modulation for communications simulations.
%   Y = MODULATE(X,Fc,Fs,METHOD,OPT) modulates the message signal X with a
%   carrier frequency Fc and sampling frequency Fs, using the modulation
%   scheme in METHOD.  OPT is an extra sometimes optional parameter whose
%   purpose depends on the modulation scheme you choose.
%
%   Fs must satisfy Fs > 2*Fc + BW, where BW is the bandwidth of the
%   modulated signal.
%
%        METHOD              MODULATION SCHEME
%    'am',      Amplitude modulation, double side-band, suppressed carrier
%    'amdsb-sc' OPT not used.
%    'amdsb-tc' Amplitude modulation, double side-band, transmitted carrier
%               OPT is a scalar which is subtracted from X prior to
%               multiplication by the carrier cosine.  It defaults to
%               min(min(X)) so the offset message signal is positive and
%               has a minimum value of zero.
%    'amssb'    Amplitude modulation, single side-band
%               OPT not used.
%    'fm'       Frequency modulation
%               OPT is a scalar which specifies the constant of frequency
%               modulation kf.  kf = (Fc/Fs)*2*pi/max(max(abs(X))) by
%               default for a maximum frequency excursion of Fc Hertz.
%    'pm'       Phase modulation
%               OPT is a scalar which specifies the constant of phase
%               modulation kp.  kp = pi/max(max(abs(x))) by default for a
%               maximum phase excursion of +/-pi radians.
%    'pwm'      Pulse width modulation
%               If you let OPT = 'centered', the pulses are centered on the
%               carrier period rather than being "left justified".
%    'ppm'      Pulse position modulation
%               OPT is a scalar between 0 and 1 which specifies the pulse
%               width in fractions of the carrier period. It defaults to .1.
%    'qam'      Quadrature amplitude modulation
%               OPT is a matrix the same size as X which is modulated in
%               quadrature with X.
%
%   If X is a matrix, its columns are modulated.
%
%   [Y,T] = MODULATE(...) returns a time vector the same length as Y.
%
%   % Example:
%   %   Frequency modulate a sinusoid with carrier frequency of 3KHz.
%
%   Fs = 8000;                              % sampling frequency
%   t = (0:1000-1)/Fs;                      % time vector
%   s = 4*cos(2*pi*500*t);                  % modulating signal
%   y = modulate(s,3e3,Fs,'fm',0.1);        % modulated signal
%   [ps,fs] = pwelch(s,[],0,length(s),Fs);  % PSD, modulating signal
%   [py,fy] = pwelch(y,[],0,length(y),Fs);  % PSD, modulated signal
%
%   subplot(211);
%   plot(fs/1e3,ps); grid on
%   title('Modulating signal')
%   ylabel('Amplitude(Watts)')
%
%   subplot(212);
%   plot(fy/1e3,py,'r'); grid on
%   title('Modulated Signal')
%   xlabel('Frequency(KHz)');
%   ylabel('Amplitude(Watts)')
%
%   See also DEMOD, VCO, COMM.
%   Note: COMM and COMM.MODEM are in the Communications Toolbox.

%   Copyright 1988-2024 The MathWorks, Inc.
%#codegen

narginchk(3,5)

if nargin > 3
    method = convertStringsToChars(varargin{1});
end

if nargin > 4
    P1 = convertStringsToChars(varargin{2});
end
% Cast to enforce precision rules
validateattributes(Fc,{'numeric'},{'real','scalar','finite'},mfilename,'FC',2)
Fc = double(Fc);
validateattributes(Fs,{'numeric'},{'real','scalar','finite'},mfilename,'FS',3);
Fs = double(Fs);
% Checks if X is a valid numeric data input
validateattributes(x,{'double','single'},{'2d'},mfilename,'X',1);
coder.internal.errorIf(Fc >= Fs/2,'signal:modulate:AliasedCarrier');
if nargin<4
    method = 'am';
end
coder.internal.assert(coder.internal.isConst(method),...
    'signal:codegeneration:OptionInputsMustBeConstant');
isInputRow = isrow(x);
if isempty(x)
    y = cast([],"like",x);
    t = cast([],"like",x);
    return
end
% convert row vector to column

x = signal.internal.toColIfVect(x);
len=size(x,1);
if strcmpi(method,'am') || strcmpi(method,'amdsb-sc')
    t_out = (0:1/Fs:((len-1)/Fs))';
    t_out = t_out(:,ones(1,size(x,2)));
    y_out = x.*cos(2*pi*Fc*t_out);
elseif strcmpi(method,'amdsb-tc')
    t_out = (0:1/Fs:((len-1)/Fs))';
    t_out = t_out(:,ones(1,size(x,2)));
    if nargin == 5
        % Cast to enforce precision rules
        validateattributes(P1,{'numeric'},{'real','scalar','finite'},mfilename,'OPT',5)
        offset = double(P1);
    else
        offset = min(x,[],'all');
    end
    y_out = ( x - offset ).*cos(2*pi*Fc*t_out);
elseif strcmpi(method,'amssb')
    t_out = (0:1/Fs:((len-1)/Fs))';
    t_out = t_out(:,ones(1,size(x,2)));
    y_out = x.*cos(2*pi*Fc*t_out) + imag(hilbert(x)).*sin(2*pi*Fc*t_out);
elseif strcmpi(method,'fm')
    t_out = (0:1/Fs:((len-1)/Fs))';
    t_out = t_out(:,ones(1,size(x,2)));
    if nargin == 5
        % Cast to enforce precision rules
        validateattributes(P1,{'numeric'},{'real','scalar','finite'},mfilename,'OPT',5)
        kf = double(P1);
    else
        x_max = max(abs(x),[],'all');
        if x_max > 0
            kf = (Fc/Fs)*2*pi/x_max;

            % default- maximum excursion of Fc Hertz
        else
            kf = zeros(class(x_max));
        end
    end
    y_out = cos(2*pi*Fc*t_out + kf*cumsum(x));   % rectangular integral approx
elseif strcmpi(method,'pm')
    t_out = (0:1/Fs:((len-1)/Fs))';
    t_out = t_out(:,ones(1,size(x,2)));
    if nargin == 5
        % Cast to enforce precision rules
        validateattributes(P1,{'numeric'},{'real','scalar','finite'},mfilename,'OPT',5)
        kp = double(P1);
    else
        kp = pi/max(abs(x),[],'all');   % default- maximum excursion of +/-pi rads
    end
    y_out = cos(2*pi*Fc*t_out + kp*x);
elseif strcmpi(method,'pwm')
    x_max = max(x,[],'all');
    x_min = min(x,[],'all');
    if (x_max > 1) || (x_min < 0)
        coder.internal.error('signal:modulate:InvalidRange', 'X', '[0,1]')
    end
    t_out = (0:(Fs/Fc*len)-1)';
    if nargin < 5
        P1 = 'left';
    else
     coder.internal.assert(ischar(P1)||isStringScalar(P1),'signal:modulate:InvalidParam')
     coder.internal.assert(coder.internal.isConst(P1),...
        'signal:codegeneration:OptionInputsMustBeConstant');
    end
    isCentered=strcmpi('centered',P1);
    isLeft = strcmpi('left',P1);
    coder.internal.assert(isCentered||isLeft,'signal:modulate:InvalidParam')
    if isCentered
        x2=[x;zeros(1,size(x,2))];
        % Class(y) must be the same as  class(x) to enforce precision rules
        y_out = zeros(length(t_out),size(x,2),class(x));  %#ok<*ZEROLIKE>
        mod_t =  rem(t_out,(Fs/Fc))/(Fs/Fc);
        ind = floor(t_out*Fc/Fs+1);
        for i = 1:size(x,2)   % for each column ...
            y_out(:,i) = (mod_t < x2(ind,i)/2)  +  ((1-mod_t) <= x2(ind+1,i)/2);
        end
    else 
        % Class(y) must be the same as  class(x) to enforce precision rules
        y_out = zeros(length(t_out),size(x,2),class(x));
        for i = 1:size(x,2)   % for each column ...
            y_out(:,i) = rem(t_out,(Fs/Fc))/(Fs/Fc) < x(floor(t_out*Fc/Fs+1),i);
        end
    end
    t_out = t_out/Fs;
elseif strcmpi(method,'ptm') || strcmpi(method,'ppm')
    x_max = max(x,[],'all');
    x_min = min(x,[],'all');
    if (x_max > 1) || (x_min < 0)
        coder.internal.error('signal:modulate:InvalidRange', 'X', '[0,1]')
    end
    if nargin<5, P1 = .1; end   % default width

    if (P1 >= 1) || (P1 <= 0)
        coder.internal.error('signal:modulate:InvalidPDRange')
    end

    if (x_max > 1-P1)
        coder.internal.warning('signal:modulate:PulseOverlapped')
    end
    % Cast to enforce precision rules
     validateattributes(P1,{'numeric'},{'scalar','finite'},mfilename,'OPT',5)
     P1 = double(P1);
    t_out = (0:(len*Fs/Fc)-1)'/Fs;
    % Class(y) must be the same as  class(x) to enforce precision rules
    y_out = zeros(length(t_out)+ceil(P1*Fs/Fc),size(x,2),class(x));
    temp = 0:(P1*Fs/Fc - 1);
    for i=1:size(x,2)    % for each column ...
        ind = ceil( (x(:,i)+(0:len-1)')*Fs/Fc )+1;
        ind = ind(:,ones(1,floor(P1*Fs/Fc) )) + temp(ones(1,length(ind)),:);
        y_out(ind(:),i) = ones(size(ind(:)));
    end
    y_out(length(t_out)+1:length(y_out),:) = []; % truncate over-long vector
elseif strcmpi(method,'qam')

    coder.internal.errorIf(nargin<5,'signal:modulate:SignalErr')
    % Checks if X and OPT are valid numeric data inputs
    validateattributes(P1,{'double','single'},{'2d'},mfilename,'OPT',5)
    x2 = P1;
    r2 = size(x2, 1);
    if r2 == 1
        x3 = x2.';
    else
        x3=x2;
    end
    coder.internal.errorIf(~isequal(size(x),size(x3)),'signal:modulate:InvalidDimensions')
    t_out = (0:1/Fs:((len-1)/Fs))';
    t_out = t_out(:,ones(1,size(x,2)));
    y_out = x.*cos(2*pi*Fc*t_out) + x3.*sin(2*pi*Fc*t_out);
else
    coder.internal.errorIf(true,'signal:modulate:InvalidMethod')
end

t_out = t_out(:,1);   % only want first column
if isInputRow   % convert y from a column to a row
    y = y_out.';
    t_in = t_out.';
else
    y = y_out;
    t_in = t_out;
end
if nargout == 2
    if isa(y_out,'single')
        t = single(t_in);
    else
        t = t_in;
    end
end

