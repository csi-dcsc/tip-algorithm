%% TIP Function Definitions

function result = ft(a)
    A = fftshift(fft2(fftshift(a)));
    r = sum(sum(abs(a))) .^ 2 ./ sum(sum(abs(A))) .^ 2;
    result = r * A;
end

function result = ift(a)  
    A = ifftshift(ifft2(ifftshift(a)));
    r = sum(sum(abs(a))) .^ 2 ./ sum(sum(abs(A))) .^ 2;
    result = r * A;
end

function result = least_squares(H,F,eps)  
    A = zeros(N, M);
    B = zeros(N, M);
    for d=1:D
        A = A + conj(H(d)) .* F(d);
        B = B + abs(H(d)) .^ 2;
    end
    B = B + eps;
    result = A ./ B
end

function result = realize(F)  
    f = real(ft(F  * filter));
    f = f ./ max(max(f));
    f = (f * (f >= lb) * (f <= ub));
    result = ift(f)
end