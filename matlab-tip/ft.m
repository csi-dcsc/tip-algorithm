%% TIP Function Definitions
function result = ft(a)
    A = fftshift(fft2(fftshift(a)));
    r = sum(sum(abs(a))) .^ 2 ./ sum(sum(abs(A))) .^ 2;
    result = r * A;
end

