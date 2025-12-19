function Cn = center_minmax(C, lower, upper)
    lower = reshape(lower,1,[]);
    upper = reshape(upper,1,[]);
    span  = upper - lower;
    span(abs(span) < eps) = 1;
    Cn = 2*(C - lower) ./ span - 1;
end

function C = center_denorm(Cn, lower, upper)
    lower = reshape(lower,1,[]);
    upper = reshape(upper,1,[]);
    span  = upper - lower;
    C = (Cn + 1)/2 .* span + lower;
    C = min(max(C, lower), upper);
end
