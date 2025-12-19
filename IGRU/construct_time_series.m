function [feat, stds, AllPCs, TimeSeries] = construct_time_series(decs, Problem, K, TimeSeries)
% 生成当前时刻特征feat（K*M*D x 1），
% stds: K x D，每个聚类全部主成分的方差，
% TimeSeries：历史特征，
% AllPCs: K×1 cell，每个为D×n_k（n_k为主成分数，最多D）

    D = Problem.D;
    M = Problem.M;
    
    % 1. PCA降维做K-means聚类
    [~, score] = pca(decs);
    % score_reduced = score(:, 1:(M-1));
    % opts = statset('MaxIter', 1000, 'UseParallel', false, 'Display','off');
    % [idxs, ~] = kmeans(score_reduced, K, 'Replicates', 5, 'Options', opts);
    score_reduced = score(:, 1:M);
    % opts = statset('MaxIter', 2000, 'UseParallel', false, 'Display','off');
    % [idxs, ~] = kmeans(score_reduced, K, 'Replicates', Problem.N, 'Options', opts);
    maxIter = Problem.N*3;
    opts = statset('MaxIter', maxIter, 'UseParallel', false, 'Display','off');
    [idxs, ~] = kmeans(score_reduced, K, ...
    'Start','plus', 'Replicates', round(Problem.N/5), ...
    'MaxIter', maxIter, 'OnlinePhase','off', ...
    'EmptyAction','singleton', 'Options', opts);
    % counts = histcounts(idxs, 1:K+1);
    % disp(counts); % 显示每组人数

    % 2. 每个聚类：提取质心+主成分向量+全部主成分方差+所有主成分方向
    Centers = zeros(K, D);        % 质心
    PCs = cell(K,1);              % 前M-1主成分（D×(M-1)）
    stds = zeros(K, D);           % 所有主成分方差
    AllPCs = cell(K,1);           % 所有主成分向量（D×n_k）
    for k = 1:K
        members = decs(idxs==k, :);
        if isempty(members)
            members = decs(randi(size(decs,1)), :);
        end
        [coeff_k, ~, latent_k, ~, ~, mu_k] = pca(members);
        Centers(k,:) = mu_k;
        % 前M-1主成分方向
        if size(coeff_k,2) >= (M-1)
            PCs{k} = coeff_k(:,1:(M-1));
        else
            PCs{k} = eye(D, M-1);
        end
        % 所有主成分方向
        AllPCs{k} = coeff_k; % D × n_k
        % 所有主成分方差
        stds(k,1:length(latent_k)) = latent_k(:)';
    end

    % 质心归一化
    % Centers = center_minmax(Centers, Problem.lower, Problem.upper);
    lowerRow = reshape(Problem.lower,1,[]);
    upperRow = reshape(Problem.upper,1,[]);
    span     = upperRow - lowerRow;
    span(abs(span) < eps) = 1;                  % 防止除零
    Centers = 2*(Centers - lowerRow) ./ span - 1;   % K x D

    % 3. 首次写入
    if isempty(TimeSeries)
        feat = [];
        for k = 1:K
            feat = [feat; Centers(k,:)'; PCs{k}(:)];
        end
        TimeSeries = feat;
        return;
    end
    
    % 4. 历史解包
    LastFeature = TimeSeries(:, end);
    OldCenters = zeros(K, D);
    OldPCs = cell(K, 1);
    offset = 0;
    for k = 1:K
        seg = LastFeature(offset + (1:(M*D)));
        OldCenters(k,:) = seg(1:D)';
        OldPCs{k} = reshape(seg(D+1:end), D, M-1);
        offset = offset + M*D;
    end
    
    % 5. munkres匹配
    Dist = pdist2(Centers, OldCenters);
    [assignment, ~] = munkres(Dist);
    [newOrder, oldOrder] = find(assignment);
    Centers = Centers(newOrder, :);
    PCs = PCs(newOrder);
    stds = stds(newOrder, :);
    AllPCs = AllPCs(newOrder);
    
    % 6. 主成分方向一致性
    for k = 1:K
        for m = 1:(M-1)
            cur_vec = PCs{k}(:,m);
            old_vec = OldPCs{oldOrder(k)}(:,m);
            if dot(cur_vec, old_vec) < 0
                PCs{k}(:,m) = -cur_vec;
            end
        end
    end

    % 7. 拼接本时刻特征：中心先归一化到[-1,1]（仅用于NN）
    % 拼接本时刻特征
    feat = [];
    for k = 1:K
        feat = [feat; Centers(k,:)'; PCs{k}(:)];
    end
    
    % 8. 更新TimeSeries
    TimeSeries = [TimeSeries, feat];
end

function Cn = center_minmax(C, lower, upper)
    lower = reshape(lower,1,[]);
    upper = reshape(upper,1,[]);
    span  = upper - lower;
    span(abs(span) < eps) = 1;                 % 防止除零
    Cn = 2*(C - lower) ./ span - 1;            % [-1,1]
end

