function result = least_squares(H,F,eps)
    [N,M,D] = size(H);
    A = zeros(N, M);
    B = zeros(N, M);
    for d=1:D
        A = A + conj(H(:,:,d)) .* F(:,:,d);
        B = B + abs(H(:,:,d)) .^ 2;
    end
    result = A ./ (B + eps);
end