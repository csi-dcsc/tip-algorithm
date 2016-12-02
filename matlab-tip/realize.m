function result = realize(F,filter,lb,ub)  
    f = real(ft(F  * filter));
    f = f ./ max(max(f));
    f = (f .* (f >= lb) .* (f <= ub));
    result = ift(f);
end