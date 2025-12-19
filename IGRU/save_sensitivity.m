% function save_ablation(Problem, IGDs, HVs, run_time, K, alph, dir, problemName, algorithmName)
function save_sensitivity(Problem, IGDs, run_time, K, alph, dir, problemName, algorithmName)
%SAVE_RESULTS 此处显示有关此函数的摘要
%   此处显示详细说明
filename = fullfile(dir, [algorithmName '_' problemName '_M' num2str(Problem.M) '_D' num2str(Problem.D) '_taut' num2str(Problem.taut(1)) '_nt' num2str(Problem.nt(1)) '_K' num2str(K) '_a' num2str(alph) '.mat']);
% 判断文件是否存在，若存在则追加，否则新建
if exist(filename, 'file')
    % S = load(filename, 'IGDs_all', 'HVs_all', 'run_times');
    S = load(filename, 'IGDs_all', 'run_times');
    if isfield(S, 'IGDs_all')
        IGDs_all = [S.IGDs_all; IGDs];
    else
        IGDs_all = IGDs;
    end
    % if isfield(S, 'HVs_all')
    %     HVs_all = [S.HVs_all; HVs];
    % else
    %     HVs_all = HVs;
    % end
    if isfield(S, 'run_times')
        run_times = [S.run_times; run_time];
    else
        run_times = run_time;
    end
else
    IGDs_all = IGDs;
    % HVs_all = HVs;
    run_times = run_time;
end
% 保存结果，覆盖写入
% save(filename, 'IGDs_all', 'HVs_all', 'run_times');
save(filename, 'IGDs_all','run_times');
end

