classdef LTFDA1 < PROBLEM
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
        starIter;   % 从第几次迭代开始
        maxIter;    % 最大迭代次数
        t_cache     % 预先枚举t的可能会取到的值
        D_cache     % 与 t_cache 一一对应的 D(t) 值
        K;          % IGRU算法超参
        alph;
        name = 'LTFDA1';
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
            G = obj.getD(t);
            g = 1 + sum((PopDec(:,2:end)-G).^2,2);
            h = 1 - sqrt(PopObj(:,1)./g);
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
            Ab = 0.5; 
            B  = Ab*sin(0.125*pi*t);          % 慢基线：零点在 t = 8n

            % 在每个 t=8n 附近加一个平滑“跳点”
            H = 5;                             % 跳点高度
            L = 0.1;                           % 跳点宽度（持续时间）
            phi = mod(t,8);                    % 相位（周期 8）
            d   = min(phi, 8-phi);             % 到最近零点的距离
            J   = H*0.5*(1 + cos(pi*d/(L/2))) .* (d <= L/2);   % 半余弦鼓包

            B = B + J;                         % 加入跳点
            D = (sin(0.5*pi*t) + B) ./ (1 + abs(B));  % 保证整体值域在 [-1,1]
        end
        %% Generate points on the Pareto front
        function R = GetOptimum(obj,N)
            R(:,1) = linspace(0,1,N)';
            R(:,2) = 1 - R(:,1).^0.5;
        end
        %% Generate the image of Pareto front
        function R = GetPF(obj)
            R = obj.GetOptimum(100);
        end
        %% Calculate the metric value
        function score = CalMetric(obj,metName,Population)                  % 迭代n次，评估是否结束迭代n+1次，计算每次评估种群的指标
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            G = obj.getD(t);
            G      = round(G*1e6)/1e6;
            change = [0;find(G(1:end-1)~=G(2:end));length(G)];              % 在Poplutaion哪些位置环境发生了改变
            Scores = zeros(1,length(change)-1);
            for i = 1 : length(change)-1
                subPop    = Population(change(i)+1:change(i+1));
                Scores(i) = feval(metName,subPop,obj.optimum);
            end
            score = mean(Scores);
        end
        function score = CalMetric2(obj,metName,Population)                  % 迭代n次，评估是否结束迭代n+1次，计算每次评估种群的指标
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            G = obj.getD(t);
            G      = round(G*1e6)/1e6;
            change = [0;find(G(1:end-1)~=G(2:end));length(G)];              % 在Poplutaion哪些位置环境发生了改变
            Scores = zeros(1,length(change)-1);
            for i = 1 : length(change)-1
                subPop    = Population(change(i)+1:change(i+1));
                Scores(i) = feval(metName,subPop,obj.optimum);
            end
            score = mean(Scores);
            if length(change) > 2
                score = Scores;
            end
        end
        %% Display a population in the objective space
        function DrawObj(obj,Population)
            t      = floor(Population.adds/obj.N/obj.taut)/obj.nt;
            G = obj.getD(t);
            G      = round(G*1e6)/1e6;
            change = [0;find(G(1:end-1)~=G(2:end));length(G)];
            tempStream = RandStream('mlfg6331_64','Seed',2);
            for i = 1 : length(change)-1
                color = rand(tempStream,1,3);
                Draw(Population(change(i)+1:change(i+1)).objs+(i-1)*0.1,'o','MarkerSize',5,'Marker','o','Markerfacecolor',sqrt(color),'Markeredgecolor',color,{'\it f\rm_1','\it f\rm_2',[]});
                Draw(obj.PF+(i-1)*0.1,'-','LineWidth',1,'Color',color);
            end
        end
    end
end
