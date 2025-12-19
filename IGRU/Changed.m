function changed = Changed(Problem,Population)
% Detect whether the problem changes

%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    RePop1  = Population(randperm(end,ceil(end/10)));
    RePop2  = Problem.Evaluation(RePop1.decs);
    % changed = ~isequal(RePop1.objs,RePop2.objs) || ~isequal(RePop1.cons,RePop2.cons);

    % 设置比较浮点数的容忍度
    tol = 1e-8;

    % 计算 obj 和 cons 的绝对差
    diffObj  = abs(RePop1.objs - RePop2.objs);
    diffCons = abs(RePop1.cons - RePop2.cons);

    % 如果任意差值超过 tol，就认为发生了变化
    changedObj  = any(diffObj(:)  > tol);
    changedCons = any(diffCons(:) > tol);
    changed = changedObj || changedCons;

    Problem.FE = Problem.FE - ceil(size(Population.decs, 1) / 10);          % 检测所消耗的评估次数不算
end