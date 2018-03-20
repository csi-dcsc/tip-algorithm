%% TIP Algorithm 
% Matlab Version 1.0
% Dean Wilding, Oleg Soloviev (c) 2016-17
% Available freely under the terms of GNU GPL


%% Initialization
clear all;

% Algorithm Parameters
N = 512;
M = 512;
D = 4;
nIter = 10;
psf_size = 15;
compression = 1;
eps = 1e-15;

% Load images
z = zeros(N,M,D);
for d=0:D-1
    img = im2double(imread(sprintf('./inputs/mountain/%d.tif',d)));
    img = img ./ sum(sum(img));
    z(:,:,d+1) = img;
end

% Apply "compression"
z_copy = z;
z = z(uint64(N/2-N/2/compression)+1:uint64(N/2+N/2/compression),uint64(M/2-M/2/compression)+1:uint64(M/2+M/2/compression),:);

N = uint64(N / compression);
M = uint64(M / compression);

% Load object
obj = im2double(imread('./inputs/mountain/object.tif'));
obj = obj ./ sum(sum(obj));

% Spatial grid
x = linspace(-1.0,1.0,N);
y = linspace(-1.0,1.0,M);
[X,Y] = meshgrid(x,y);
RHO = sqrt(X.^2+Y.^2);

% Create basis functions (these may be saved to disk)
mask = (RHO < (double(psf_size) / double(N))) .* ones(N,M);
K = uint64(sum(sum(mask)));

A = zeros(M * N, K);
for i=1:K
    f = zeros(N, M);
    [maxX, idxX] = max(mask);
    [maxY, idxY] = max(maxX);
    px = idxX(idxY);
    py = idxY;
    f(px, py) = 1;
    mask(px, py) = 0;
    F = ift(f);
    A(:, i) = reshape(F,M * N,1);
end

% Invert the matrix
Ainv = pinv(A);

% Compute the OTFs and apply the filters
% Generate starting OTFs - blind
H = ones(N, M, D);
Z = zeros(N, M, D);

for d=1:D
    Z(:,:,d) = ift(z(:,:,d));
end

%% Main algorithm loop
for i=1:nIter
    O = least_squares(H, Z, eps);
    % Renormalisation
    O = real(ft(O));
    O( O < 0 ) = 0;
    O = O ./ sum(sum(abs(O)));
    O = ift(O);
    for d=1:D
        H(:,:,d) = Z(:,:,d) ./ (O + eps);
        alpha = real(Ainv * reshape(H(:,:,d),M*N,1));
        alpha(alpha < 0) = 0;
        % Renormalisation
        alpha = alpha / sum(sum(abs(alpha)));
        H(:,:,d) = reshape(A * alpha,N,M);
    end
end

%% PSFs
N = uint64(N*compression);
M = uint64(M*compression);
h = zeros(N,M,D);
for d=1:D
    h(uint64(N/2-N/2/compression)+1:uint64(N/2+N/2/compression),uint64(M/2-M/2/compression)+1:uint64(M/2+M/2/compression),d) = real(ft(H(:,:,d)));
end

%% Final step: final deconvolution (full-size)
z = z_copy;
Z = zeros(N, M, D);
H = zeros(N, M, D);
for d=1:D
    Z(:,:,d) = ift(z(:,:,d));
    H(:,:,d) = ift(h(:,:,d));
end

O = least_squares(H, Z, eps);
o = real(ft(O));

% Threshold and normalization
o(o<0) = 0;
o = o / max(max(o));

for d=1:D
    h(:,:,d) = h(:,:,d) ./ max(max(h(:,:,d)));
    h(:,:,d) = h(:,:,d) .* (h(:,:,d) >= 0.0);
end

%% Display Outputs
% Images
figure;
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
imshow(log(abs(O)/max(max(abs(O)))),[]);
title('O Spectra Amp.')
subplot(3,4,10)
imshow(angle(O),[]);
title('O Spectra Phase')
subplot(3,4,11)
imshow(o, [0 1.0]);
subplot(3,4,12)
title('TIP Object')
imshow(obj, [0 1.0]);
title('Real Object')

figure;
imshow(o, [0 1.0]);

