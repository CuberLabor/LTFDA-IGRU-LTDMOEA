%%  Modified linear prediction model
function newPopS = Response_Short(Population,N,Curr_C,Last_C,Problem)
    d=Curr_C-Last_C;
    New_Pop1_decs=Population.decs;
    for i=1:N
        step = randsample([0.5, 1, 1.5], 1);
        % New_Pop1_decs(i,:)=New_Pop1_decs(i,:) + step * d + sqrt(sigma) * rand(1,D);
        New_Pop1_decs(i,:)=New_Pop1_decs(i,:) + step * d;
        New_Pop1_decs(i,:)= min(max(New_Pop1_decs(i,:),Problem.lower),Problem.upper);
    end
    newPopS=Problem.Evaluation(New_Pop1_decs);
end
