classdef LTFDA2 < PROBLEM
% <multi> <real> <large/none> <dynamic>
% Benchmark dynamic MOP proposed by Farina, Deb, and Amato
% taut --- 10 --- Number of generations for static optimization
% nt   --- 10 --- Number of distinct steps
% starIter --- 0 --- 从第几次种群迭代次数开始
% maxIter --- 100 --- 种群最大迭代次数

%------------------------------- Reference --------------------------------
% M. Farina, K. Deb, and P. Amato, Dynamic multiobjective optimization
% problems: Test cases, approximations, and applications, IEEE Transactions
% on Evolutionary Computation, 2004, 8(5): 425-442.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
	
    properties
        taut;       % Number of generations for static optimization
        nt;         % Number of distinct steps
        Optimums;   % Point sets on all Pareto fronts
        starIter;   % 从第几次迭代开始
        maxIter;    % 最大迭代次数
        t_cache     % 预先枚举t的可能会取到的值
        D_cache     % 与 t_cache 一一对应的 D(t) 值
        K;          % IGRU算法超参
        alph;
        name = 'LTFDA2';
    end
    methods
       
        %% Default settings of the problem
        function Setting(obj)
            % [obj.taut,obj.nt,obj.starT,obj.maxT] = obj.ParameterSet(50,10,0,1000);
            % obj.maxFE = obj.N * obj.taut * (obj.maxT+1);                        % 要马上依据最大环境变化次数修正最大评估次数
            [obj.taut,obj.nt,obj.starIter,obj.maxIter] = obj.ParameterSet(10,10,0,4800);
            % [obj.taut,obj.nt,obj.starIter,obj.maxIter,obj.K,obj.alph] = obj.ParameterSet(10,10,0,4800,5,0.1);
            obj.maxFE = obj.N * obj.maxIter;
            obj.M = 2;
            if isempty(obj.D); obj.D = 10; end
            obj.D        = ceil((obj.D-1)/2)*2 + 1;
            obj.lower    = [0,-ones(1,obj.D-1)];
            obj.upper    = [1, ones(1,obj.D-1)];
            obj.encoding = ones(1,obj.D);
            obj.t_cache = (0:floor(obj.maxIter / obj.taut)) ./ obj.nt;      % 所有可能出现的t值
            obj.D_cache = obj.raw_getD(obj.t_cache);                        % 将与t对应的D(t)一次性算好
        end
        %% Evaluate solutions
        function Population = Evaluation(obj,varargin)
            PopDec     = obj.CalDec(varargin{1});
            PopObj     = obj.CalObj(PopDec);
            PopCon     = obj.CalCon(PopDec);
            % Attach the current number of function evaluations to solutions
            Population = SOLUTION(PopDec,PopObj,PopCon,zeros(size(PopDec,1),1)+obj.FE);
            obj.FE     = obj.FE + length(Population);
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            PopObj(:,1) = PopDec(:,1); 
            t = floor(obj.FE/obj.N/obj.taut)/obj.nt;
            g = 1 + sum(PopDec(:,2:(end+1)/2).^2,2);
            % H = 0.75 + 0.7*sin(0.5.*pi.*t);
            H = obj.getD(t);
            % Note: The original definition of h is questionable
            h = 1 - (PopObj(:,1)./g).^(H+sum((PopDec(:,(end+1)/2+1:end)-H).^2,2));
            PopObj(:,2) = g.*h;
        end
        %% 得到动态因子
        function D = getD(obj,t)
            % 查表：t 理论上总是 s/nt 的离散值，这里做四舍五入配对
            if isempty(obj.t_cache) || isempty(obj.D_cache)
                % 保险起见：若未初始化缓存，则退回原公式计算
                D = obj.raw_getD(t);
                return;
            end
            D = obj.D_cache(round(t .* obj.nt)+1)';
        end
        function D = raw_getD(obj,t)
            Ab = 0.3;                         % 基线幅度（确保 <= 0.75）
            B  = Ab * sin(0.125*pi*t);

            % 规律性突变 J：每 8 个单位一次，半余弦“鼓包”，宽度 L
            H  = 0.6;                         % 跳点高度（确保 |B+J| <= 0.75）
            L  = 0.1;                         % 持续时间
            phi = mod(t, 8);
            d   = min(phi, 8 - phi);          % 到最近 t=8n 的距离
            J   = H*0.5*(1 + cos(pi*d/(L/2))) .* (d <= L/2);

            % 合成慢变化项 C，并可选“安全夹紧”避免超界
            C = B + J;
            % C = max(min(C, 0.75), -0.75);     % 可选：确保 |C| <= 0.75

            % 幅度自收缩：A(t) = 0.75 - |C(t)|，中心移到 0.75 + C(t)
            A = max(0, 0.75 - abs(C));
            D = (0.75 + C) + A .* sin(0.5*pi * t);   % 值域严格落在 [0, 1.5]
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            % Generate point sets on all Pareto fronts
            t = floor((0:obj.maxFE)/obj.N/obj.taut)/obj.nt;
            % H = 0.75 + 0.7*sin(0.5.*pi.*t);
            H = obj.getD(t);
            H = unique(round(H*1e6)/1e6);
            x = linspace(0,1,N)';
            obj.Optimums = {};
            for i = 1 : length(H)
                obj.Optimums(i,:) = {H(i),[x,1-x.^(H(i)+max(0,H(i)-1).^2*(obj.D-1)/2)]};
            end
            % Combine all point sets
            R = cat(1,obj.Optimums{:,2});
        end
        %% Calculate the metric value
        function score = CalMetric(obj,metName,Population)
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            % H      = 0.75 + 0.7*sin(0.5.*pi.*t);
            H = obj.getD(t);
            H      = round(H*1e6)/1e6;
            change = [0;find(H(1:end-1)~=H(2:end));length(H)];
            Scores = zeros(1,length(change)-1);
            allH   = cell2mat(obj.Optimums(:,1));
            for i = 1 : length(change)-1
                subPop    = Population(change(i)+1:change(i+1));
                % Scores(i) = feval(metName,subPop,obj.Optimums{find(H(change(i)+1)==allH,1),2});
                tolerance = 1e-10; % 设置一个小的容差
                find_H_idex = find(abs(H(change(i)+1) - allH) < tolerance, 1);
                Scores(i) = feval(metName,subPop,obj.Optimums{find_H_idex,2});
            end
            score = mean(Scores);
        end
        function score = CalMetric2(obj,metName,Population)
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            % H      = 0.75 + 0.7*sin(0.5.*pi.*t);
            H = obj.getD(t);
            H      = round(H*1e6)/1e6;
            change = [0;find(H(1:end-1)~=H(2:end));length(H)];
            Scores = zeros(1,length(change)-1);
            allH   = cell2mat(obj.Optimums(:,1));
            for i = 1 : length(change)-1
                subPop    = Population(change(i)+1:change(i+1));
                % Scores(i) = feval(metName,subPop,obj.Optimums{find(H(change(i)+1)==allH,1),2});
                tolerance = 1e-10; % 设置一个小的容差
                find_H_idex = find(abs(H(change(i)+1) - allH) < tolerance, 1);
                Scores(i) = feval(metName,subPop,obj.Optimums{find_H_idex,2});
            end
            score = mean(Scores);
            if length(change) > 2
                score = Scores;
            end
        end
        %% Display a population in the objective space
        function DrawObj(obj,Population)
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            % H      = 0.75 + 0.7*sin(0.5.*pi.*t);
            H = obj.getD(t);
            H      = round(H*1e6)/1e6;
            change = [0;find(H(1:end-1)~=H(2:end));length(H)];
            allH   = cell2mat(obj.Optimums(:,1));
            tempStream = RandStream('mlfg6331_64','Seed',2);
            for i = 1 : length(change)-1
                color = rand(tempStream,1,3);
                Draw(Population(change(i)+1:change(i+1)).objs,'o','MarkerSize',5,'Marker','o','Markerfacecolor',sqrt(color),'Markeredgecolor',color,{'\it f\rm_1','\it f\rm_2',[]});
                % Draw(obj.Optimums{find(H(change(i)+1)==allH,1),2},'-','LineWidth',1,'Color',color);
                tolerance = 1e-10; % 设置一个小的容差
                find_H_idex = find(abs(H(change(i)+1) - allH) < tolerance, 1);
                Draw(obj.Optimums{find_H_idex,2},'-','LineWidth',1,'Color',color);
            end
        end
    end
end
