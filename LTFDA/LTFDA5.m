classdef LTFDA5 < PROBLEM
% <multi/many> <real> <large/none> <dynamic>
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
        name = 'LTFDA5';
    end
    methods
        %% Default settings of the problem
        function Setting(obj)
            % [obj.taut,obj.nt,obj.starIter,obj.maxIter] = obj.ParameterSet(10,10,0,100);
            % obj.maxFE = obj.N * obj.taut * (obj.maxT+1);                        % 要马上依据最大环境变化次数修正最大评估次数
            [obj.taut,obj.nt,obj.starIter,obj.maxIter] = obj.ParameterSet(10,10,0,4800);
            % [obj.taut,obj.nt,obj.starIter,obj.maxIter,obj.K,obj.alph] = obj.ParameterSet(10,10,0,4800,5,0.1);
            obj.maxFE = obj.N * obj.maxIter;
            if isempty(obj.M); obj.M = 3; end
            if isempty(obj.D); obj.D = obj.M+9; end
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
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
            t = floor(obj.FE/obj.N/obj.taut)/obj.nt;
            PopDec(:,1:obj.M-1) = PopDec(:,1:obj.M-1).^(1+100*sin(0.5*pi*t).^4);
            % G = abs(sin(0.5.*pi.*t));
            G = obj.getD(t);
            g = G + sum((PopDec(:,obj.M:end)-G).^2,2);
            PopObj = repmat(1+g,1,obj.M).*fliplr(cumprod([ones(size(g,1),1),cos(PopDec(:,1:obj.M-1)*pi/2)],2)).*[ones(size(g,1),1),sin(PopDec(:,obj.M-1:-1:1)*pi/2)];
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
            % 基信号
            base = abs(sin(0.5*pi * t));

            % 慢基线 B（例：周期 16）
            Ab = 0.25;
            B  = 0.25 + Ab * sin(0.125*pi*t);

            % 规律突变 J（例：每 T=8 出现一次，半余弦平滑鼓包，宽度 L）
            H = 0.75; T = 8; L = 0.1;
            phi = mod(t, T);
            d   = min(phi, T - phi);
            J   = H * 0.5 * (1 + cos(pi * d / (L/2))) .* (d <= L/2);

            % 合成慢变化项
            C = B + J;
            D=(C + base)/(1+ 0.25 + 0.25*sin(3*pi/8));
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            R = UniformPoint(N,obj.M);
            R = R./repmat(sqrt(sum(R.^2,2)),1,obj.M);
            % Generate point sets on all Pareto fronts
            t = floor(0:obj.maxFE/obj.N/obj.taut)/obj.nt;
            % G = abs(sin(0.5.*pi.*t));
            G = obj.getD(t);
            G = unique(round(G*1e6)/1e6);
            obj.Optimums = {};
            for i = 1 : length(G)
                obj.Optimums(i,:) = {G(i),R.*(1+G(i))};
            end
            % Combine all point sets
            R = cat(1,obj.Optimums{:,2});
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            if obj.M == 2
                R = UniformPoint(100,2);
                R = R./repmat(sqrt(sum(R.^2,2)),1,2);
            elseif obj.M == 3
                a = linspace(0,pi/2,10)';
                R = {sin(a)*cos(a'),sin(a)*sin(a'),cos(a)*ones(size(a'))};
            else
                R = [];
            end
        end
        %% Calculate the metric value
        function score = CalMetric(obj,metName,Population)
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            % G      = abs(sin(0.5.*pi.*t));
            G = obj.getD(t);
            G      = round(G*1e6)/1e6;
            change = [0;find(G(1:end-1)~=G(2:end));length(G)];
            Scores = zeros(1,length(change)-1);
            allG   = cell2mat(obj.Optimums(:,1));
            for i = 1 : length(change)-1
                subPop    = Population(change(i)+1:change(i+1));
                % Scores(i) = feval(metName,subPop,obj.Optimums{find(G(change(i)+1)==allG,1),2});
                tolerance = 1e-10; % 设置一个小的容差
                find_G_idex = find(abs(G(change(i)+1) - allG) < tolerance, 1);
                Scores(i) = feval(metName,subPop,obj.Optimums{find_G_idex,2});
            end
            score = mean(Scores);
        end
        function score = CalMetric2(obj,metName,Population)
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            % G      = abs(sin(0.5.*pi.*t));
            G = obj.getD(t);
            G      = round(G*1e6)/1e6;
            change = [0;find(G(1:end-1)~=G(2:end));length(G)];
            Scores = zeros(1,length(change)-1);
            allG   = cell2mat(obj.Optimums(:,1));
            for i = 1 : length(change)-1
                subPop    = Population(change(i)+1:change(i+1));
                % Scores(i) = feval(metName,subPop,obj.Optimums{find(G(change(i)+1)==allG,1),2});
                tolerance = 1e-10; % 设置一个小的容差
                find_G_idex = find(abs(G(change(i)+1) - allG) < tolerance, 1);
                Scores(i) = feval(metName,subPop,obj.Optimums{find_G_idex,2});
            end
            score = mean(Scores);
            if length(change) > 2
                score = Scores;
            end
        end
        %% Display a population in the objective space
        function DrawObj(obj,Population)
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            % G      = abs(sin(0.5.*pi.*t));
            G = obj.getD(t);
            G      = round(G*1e6)/1e6;
            change = [0;find(G(1:end-1)~=G(2:end));length(G)];
            tempStream = RandStream('mlfg6331_64','Seed',2);
            if obj.M == 2
                for i = 1 : length(change)-1
                    color = rand(tempStream,1,3);
                    Draw(Population(change(i)+1:change(i+1)).objs,'o','MarkerSize',5,'Marker','o','Markerfacecolor',sqrt(color),'Markeredgecolor',color,{'\it f\rm_1','\it f\rm_2',[]});
                    Draw(obj.PF*(1+G(change(i)+1)),'-','LineWidth',1,'Color',color);
                end
            elseif obj.M == 3
                for i = 1 : length(change)-1
                    color = rand(tempStream,1,3);
                    ax = Draw(Population(change(i)+1:change(i+1)).objs,'o','MarkerSize',6,'Marker','o','Markerfacecolor',sqrt(color),'Markeredgecolor',color,{'\it f\rm_1','\it f\rm_2','\it f\rm_3'});
                    surf(ax,obj.PF{1}*(1+G(change(i)+1)),obj.PF{2}*(1+G(change(i)+1)),obj.PF{3}*(1+G(change(i)+1)),'EdgeColor',color,'FaceColor','none');
                end
            else
                for i = 1 : length(change)-1
                    Draw(Population(change(i)+1:change(i+1)).objs,'-','Color',rand(tempStream,1,3),'LineWidth',2);
                end
            end
        end
    end
end
