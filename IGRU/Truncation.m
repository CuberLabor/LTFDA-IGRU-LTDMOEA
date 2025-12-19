function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation
% 依次从PopObj中删除最拥挤的K个解，返回的Del是要删除解的位置

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Truncation
    Distance = pdist2(PopObj,PopObj);
    % 将对角线上的距离(即自己到自己的距离)设为无穷大
    Distance(logical(eye(length(Distance)))) = inf;
    % 初始化删除标记向量,长度为种群大小
    Del = false(1,size(PopObj,1));
    % 循环直到删除K个解
    while sum(Del) < K
        % 找出还未被删除的解的索引
        Remain   = find(~Del);
        % 对所有未删除的解的距离排序
        Temp     = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        % 将找到的最拥挤的解标记为删除
        Del(Remain(Rank(1))) = true;
    end
end

