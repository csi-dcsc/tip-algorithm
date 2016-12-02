function result = ift(a)  
    A = ifftshift(ifft2(ifftshift(a)));
    r = sum(sum(abs(a))) .^ 2 ./ sum(sum(abs(A))) .^ 2;
    result = r * A;
end
