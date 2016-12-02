%% TIP Algorithm (Basic Version)
% Matlab Version 1.0
% Dean Wilding (c) 2016
% Available freely under the terms of GNU GPL


%% Initialization
% Algorithm Parameters
N = 512;
M = 512;
D = 4;
nIter = 10;
aperture = 0.45;
lb = 0.15;
ub = 1.0;
eps = 1e-15;

% Load images
z = zeros(N,M,D);
for d=0:3
    img = im2double(imread(sprintf('inputs/mountain/%d.tif',d)));
    img = img ./ max(max(img));
    z(:,:,d+1) = img;
end

% Load object
obj = im2double(imread('inputs/mountain/object.tif'));
obj = obj ./ max(max(obj));

% Spatial grid
x = -1.0:(2.0/(N-1)):1.0;
y = -1.0:(2.0/(N-1)):1.0;
[X,Y] = meshgrid(x,y);
RHO = sqrt(X.^2+Y.^2);

% Spatial filters
filter = double((RHO < 2*aperture) .* ones(N,M));

% Compute the OTFs and apply the filters
% Generate starting OTFs - blind
H = ones(N, M, D);
Z = zeros(N, M, D);

for d=1:D
    Z(:,:,d) = ift(z(:,:,d));
    Z(:,:,d) = Z(:,:,d) .* filter;
    H(:,:,d) = H(:,:,d) .* filter;
end

% Initial Object Spectrum
O = ones(N, M);

%% Main algorithm loop
for i=1:nIter
    O = least_squares(H, Z, eps);
    for d=1:D
        H(:,:,d) = Z(:,:,d) .* conj(O) ./ (abs(O).^2 + eps);
        H(:,:,d) = realize(H(:,:,d),filter,lb,ub);
    end
end

%% Final step: final deconvolution
O = zeros(N,M);
O = least_squares(H, Z, eps) .* filter;
o = real(ft(O));
O = ift(o);
o = real(ft(O));
o = o - min(min(o));
o = o ./ max(max(o));

%% PSFs
h = zeros(N,M,D);
for d=1:D
    h(:,:,d) = real(ft(H(:,:,d)));
    h(:,:,d) = h(:,:,d) ./ max(max(h(:,:,d)));
    h(:,:,d) = h(:,:,d) .* (h(:,:,d) >= 0.0);
end

%% Display Outputs
% Images
subplot(3,4,1)
imshow(z(:,:,1), [0 1.0]);
title('Image 1')
subplot(3,4,2)
imshow(z(:,:,2), [0 1.0]);
title('Image 2')
subplot(3,4,3)
imshow(z(:,:,3), [0 1.0]);
title('Image 3')
subplot(3,4,4)
imshow(z(:,:,4), [0 1.0]);
title('Image 4')
% PSFs
subplot(3,4,5)
imshow(h(N/2-10:N/2+10,M/2-10:M/2+10,1), [0 1.0]);
title('PSF 1')
subplot(3,4,6)
imshow(h(N/2-10:N/2+10,M/2-10:M/2+10,2), [0 1.0]);
title('PSF 2')
subplot(3,4,7)
imshow(h(N/2-10:N/2+10,M/2-10:M/2+10,3), [0 1.0]);
title('PSF 3')
subplot(3,4,8)
imshow(h(N/2-10:N/2+10,M/2-10:M/2+10,4), [0 1.0]);
title('PSF 4')
% Objects
subplot(3,4,9)
imshow(log(abs(O)/max(max(abs(O)))));
title('O Spectra Amp.')
subplot(3,4,10)
imshow(angle(O));
title('O Spectra Phase')
subplot(3,4,11)
imshow(o, [0 1.0]);
subplot(3,4,12)
title('TIP Object')
imshow(obj, [0 1.0]);
title('Real Object')

