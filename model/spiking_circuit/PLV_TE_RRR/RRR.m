function RRR(varargin)

% Do reduced rank regression (RRR)
% The code in the paper "Cortical areas interact through a communication subspace", Semedo et al. (Neuron, 2019) 
% are used for this analysis (see 'https://github.com/joao-semedo/communication-subspace')
% These codes are at 'Distributed_and_dynamical_communication/analysis/sub_space'.

% Run 'onoff_detection.py', 'extractDataforRRRforEachNet.py' and 'combineDataforRRR.py' before running this script.

currentDir = pwd; 
cd('PATH_to_/Distributed_and_dynamical_communication/analysis/sub_space')  
% replace the above path with your path to 'Distributed_and_dynamical_communication/analysis/sub_space'.
addpath('.')
startup
cd(currentDir) % 


rng(1)
save_fig = 1;

numDimsUsedForPrediction = 1:10;
C_RIDGE_D_MAX_SHRINKAGE_FACTOR = .5:.01:1;

ridgeInit = 1;
topdown = 0; % do prediction in the bottom-up direction, i.e., using Area 1's spiking activity to predict Area 2's spiking activity 

scale = false;

sprintf('ridgeInit: %d',ridgeInit)

condi = 'stim';

if topdown
    rv = '_rv';
else
    rv = '';
end
sprintf('topdown: %d',topdown)

mkdir RRR_results
if ridgeInit == 0
    results_dir = ['RRR_results/', condi, '_RRR_syncUnsync_rg5',rv,'/']; %'results/';    _RRR_syncUnsync_Evt_rg8
else
    results_dir = ['RRR_results/', condi, '_ridgeRRR_syncUnsync_rg5',rv,'/']; %'results/'; _ridgeRRR_syncUnsync_Evt_rg8
end

mkdir(sprintf('%s',results_dir))

%%
if ridgeInit
    data_file_name = [condi, '_ridgeRRR_syncUnsync_sua_lc_rg5',rv,'.mat']; % stim_ridgeRRR_att_whole '_ridgeRRR_syncUnsync_sua_lc_Evt_rg8.mat'
else
    data_file_name = [condi, '_RRR_syncUnsync_sua_lc_rg5',rv,'.mat']; % stim_RRR_att_whole
end   

saveFigSuffix = ['_syncUnsync_lc_rg5',rv]; % _whole


%%

load(['raw_data/noatt_rg5_ctrsua_sync_local_subM1_comb.mat'])
load(['raw_data/noatt_rg5_ctrsua_unsync_local_subM1_comb.mat'])
load(['raw_data/att_rg5_ctrsua_sync_local_subM1_comb.mat'])
load(['raw_data/att_rg5_ctrsua_unsync_local_subM1_comb.mat'])


if topdown
    X_ctrSpk_syncCtr_noatt = a2_rg5_ctrsua_sync_noatt_lc_ctr;
    y_ctrSpk_syncCtr_noatt = a1_rg5_ctrsua_sync_noatt_lc_ctr;

    X_ctrSpk_unsyncCtr_noatt = a2_rg5_ctrsua_unsync_noatt_lc_ctr;
    y_ctrSpk_unsyncCtr_noatt = a1_rg5_ctrsua_unsync_noatt_lc_ctr;

    X_ctrSpk_syncCtr_att = a2_rg5_ctrsua_sync_att_lc_ctr;
    y_ctrSpk_syncCtr_att = a1_rg5_ctrsua_sync_att_lc_ctr;

    X_ctrSpk_unsyncCtr_att = a2_rg5_ctrsua_unsync_att_lc_ctr;
    y_ctrSpk_unsyncCtr_att = a1_rg5_ctrsua_unsync_att_lc_ctr;
else
    X_ctrSpk_syncCtr_noatt = a1_rg5_ctrsua_sync_noatt_lc_ctr;
    y_ctrSpk_syncCtr_noatt = a2_rg5_ctrsua_sync_noatt_lc_ctr;

    X_ctrSpk_unsyncCtr_noatt = a1_rg5_ctrsua_unsync_noatt_lc_ctr;
    y_ctrSpk_unsyncCtr_noatt = a2_rg5_ctrsua_unsync_noatt_lc_ctr;

    X_ctrSpk_syncCtr_att = a1_rg5_ctrsua_sync_att_lc_ctr;
    y_ctrSpk_syncCtr_att = a2_rg5_ctrsua_sync_att_lc_ctr;

    X_ctrSpk_unsyncCtr_att = a1_rg5_ctrsua_unsync_att_lc_ctr;
    y_ctrSpk_unsyncCtr_att = a2_rg5_ctrsua_unsync_att_lc_ctr;

end

X_ctrSpk_syncCtr_noatt = sub_mean(X_ctrSpk_syncCtr_noatt);
y_ctrSpk_syncCtr_noatt = sub_mean(y_ctrSpk_syncCtr_noatt);

X_ctrSpk_unsyncCtr_noatt = sub_mean(X_ctrSpk_unsyncCtr_noatt);
y_ctrSpk_unsyncCtr_noatt = sub_mean(y_ctrSpk_unsyncCtr_noatt);

X_ctrSpk_syncCtr_att = sub_mean(X_ctrSpk_syncCtr_att);
y_ctrSpk_syncCtr_att = sub_mean(y_ctrSpk_syncCtr_att);

X_ctrSpk_unsyncCtr_att = sub_mean(X_ctrSpk_unsyncCtr_att);
y_ctrSpk_unsyncCtr_att = sub_mean(y_ctrSpk_unsyncCtr_att);




%% centre spk sync ctr noatt
src = X_ctrSpk_syncCtr_noatt;
trg = y_ctrSpk_syncCtr_noatt;

[lambdaOpt, cvLoss_ridge_opt, cvLoss_ridge] = cv_RidgeReg_acrossNet(src, trg, ...
    C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale);

R.lambdaOpt_ctrSpk_syncCtr_noatt = lambdaOpt; 
R.cvLoss_ridge_opt_ctrSpk_syncCtr_noatt = cvLoss_ridge_opt ;
R.cvLoss_ridge_ctrSpk_syncCtr_noatt = cvLoss_ridge;


[optDim, cvLoss_RRR_opt, cvLoss_RRR] = cv_RRR_acrossNet(src, ...
    trg, numDimsUsedForPrediction, ridgeInit, scale);

R.optDim_ctrSpk_syncCtr_noatt = optDim; 
R.cvLoss_RRR_opt_ctrSpk_syncCtr_noatt = cvLoss_RRR_opt;
R.cvLoss_RRR_ctrSpk_syncCtr_noatt = cvLoss_RRR;


%% centre spk unsync ctr noatt
src = X_ctrSpk_unsyncCtr_noatt;
trg = y_ctrSpk_unsyncCtr_noatt;

[lambdaOpt, cvLoss_ridge_opt, cvLoss_ridge] = cv_RidgeReg_acrossNet(src, trg, ...
    C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale);

R.lambdaOpt_ctrSpk_unsyncCtr_noatt = lambdaOpt; 
R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_noatt = cvLoss_ridge_opt ;
R.cvLoss_ridge_ctrSpk_unsyncCtr_noatt = cvLoss_ridge;


[optDim, cvLoss_RRR_opt, cvLoss_RRR] = cv_RRR_acrossNet(src, ...
    trg, numDimsUsedForPrediction, ridgeInit, scale);

R.optDim_ctrSpk_unsyncCtr_noatt = optDim; 
R.cvLoss_RRR_opt_ctrSpk_unsyncCtr_noatt = cvLoss_RRR_opt;
R.cvLoss_RRR_ctrSpk_unsyncCtr_noatt = cvLoss_RRR;


%% centre spk sync ctr att
src = X_ctrSpk_syncCtr_att;
trg = y_ctrSpk_syncCtr_att;

[lambdaOpt, cvLoss_ridge_opt, cvLoss_ridge] = cv_RidgeReg_acrossNet(src, trg, ...
    C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale);

R.lambdaOpt_ctrSpk_syncCtr_att = lambdaOpt; 
R.cvLoss_ridge_opt_ctrSpk_syncCtr_att = cvLoss_ridge_opt ;
R.cvLoss_ridge_ctrSpk_syncCtr_att = cvLoss_ridge;


[optDim, cvLoss_RRR_opt, cvLoss_RRR] = cv_RRR_acrossNet(src, ...
    trg, numDimsUsedForPrediction, ridgeInit, scale);

R.optDim_ctrSpk_syncCtr_att = optDim; 
R.cvLoss_RRR_opt_ctrSpk_syncCtr_att = cvLoss_RRR_opt;
R.cvLoss_RRR_ctrSpk_syncCtr_att = cvLoss_RRR;


%% centre spk unsync ctr att
src = X_ctrSpk_unsyncCtr_att;
trg = y_ctrSpk_unsyncCtr_att;

[lambdaOpt, cvLoss_ridge_opt, cvLoss_ridge] = cv_RidgeReg_acrossNet(src, trg, ...
    C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale);

R.lambdaOpt_ctrSpk_unsyncCtr_att = lambdaOpt; 
R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_att = cvLoss_ridge_opt ;
R.cvLoss_ridge_ctrSpk_unsyncCtr_att = cvLoss_ridge;


[optDim, cvLoss_RRR_opt, cvLoss_RRR] = cv_RRR_acrossNet(src, ...
    trg, numDimsUsedForPrediction, ridgeInit, scale);

R.optDim_ctrSpk_unsyncCtr_att = optDim; 
R.cvLoss_RRR_opt_ctrSpk_unsyncCtr_att = cvLoss_RRR_opt;
R.cvLoss_RRR_ctrSpk_unsyncCtr_att = cvLoss_RRR;

%% save data
save([results_dir, data_file_name], 'R')

%%

clr2 = [255 128 0]/255; % att
clr = [[0 0.4470 0.7410]; clr2; [1 1 1]];
mksize = 8;

NetNum = numel(X_ctrSpk_syncCtr_noatt);
%%
for NetId=1:NetNum

fig1 = figure;
fig1.Position = [200,200,1200,500];

%%%     ctrSpk_syncCtr noatt
subplot1 = subplot(1,2,1);
x_plt = numDimsUsedForPrediction;
errorbar(x_plt, 1-R.cvLoss_RRR_ctrSpk_syncCtr_noatt(1,:,NetId), ...
    R.cvLoss_RRR_ctrSpk_syncCtr_noatt(2,:,NetId), ...
    'o-', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','RRR sync')
hold on

errorbar(0, 1-R.cvLoss_ridge_opt_ctrSpk_syncCtr_noatt(1,NetId), ...
    R.cvLoss_ridge_opt_ctrSpk_syncCtr_noatt(2,NetId), ...
    '^-', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','full model sync')

optDim = R.optDim_ctrSpk_syncCtr_noatt(NetId);
txt_y1 = 1 + diff(R.cvLoss_RRR_ctrSpk_syncCtr_noatt(:,optDim,NetId)) + 0.01;
text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))

%%%     ctrSpk_unsyncCtr noatt
errorbar(x_plt, 1-R.cvLoss_RRR_ctrSpk_unsyncCtr_noatt(1,:,NetId), ...
    R.cvLoss_RRR_ctrSpk_unsyncCtr_noatt(2,:,NetId), ...
    'o-', 'Color', clr(1,:), 'MarkerFaceColor', clr(3,:), 'MarkerSize', mksize, 'DisplayName','RRR unsync')
hold on

errorbar(0, 1-R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_noatt(1,NetId), ...
    R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_noatt(2,NetId), ...
    '^-', 'Color', clr(1,:), 'MarkerFaceColor', clr(3,:), 'MarkerSize', mksize, 'DisplayName','full model unsync')

optDim = R.optDim_ctrSpk_unsyncCtr_noatt(NetId);
txt_y1 = 1 + diff(R.cvLoss_RRR_ctrSpk_unsyncCtr_noatt(:,optDim,NetId)) + 0.01;
text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))


[ylim_min1, ylim_max1] = get_ylim(R.cvLoss_ridge_opt_ctrSpk_syncCtr_noatt(:,NetId), R.cvLoss_RRR_ctrSpk_syncCtr_noatt(:,:,NetId));

[ylim_min2, ylim_max2] = get_ylim(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_noatt(:,NetId), R.cvLoss_RRR_ctrSpk_unsyncCtr_noatt(:,:,NetId));


ylim(subplot1, [min(ylim_min1,ylim_min2), max(ylim_max1,ylim_max2)])
xlim([-1, max(x_plt)+1])

xlabel('number of predicitve dimensions')
ylabel('prediction performance')
title('sens > asso; centre group; noatt')

legend('location', 'best')

%%%     ctrSpk_syncCtr att
subplot1 = subplot(1,2,2);
x_plt = numDimsUsedForPrediction;
errorbar(x_plt, 1-R.cvLoss_RRR_ctrSpk_syncCtr_att(1,:,NetId), ...
    R.cvLoss_RRR_ctrSpk_syncCtr_att(2,:,NetId), ...
    'o-', 'Color', clr(2,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','RRR sync')
hold on

errorbar(0, 1-R.cvLoss_ridge_opt_ctrSpk_syncCtr_att(1,NetId), ...
    R.cvLoss_ridge_opt_ctrSpk_syncCtr_att(2,NetId), ...
    '^-', 'Color', clr(2,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','full model sync')

optDim = R.optDim_ctrSpk_syncCtr_att(NetId);
txt_y1 = 1 + diff(R.cvLoss_RRR_ctrSpk_syncCtr_att(:,optDim,NetId)) + 0.01;
text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))

%%%     ctrSpk_unsyncCtr noatt
errorbar(x_plt, 1-R.cvLoss_RRR_ctrSpk_unsyncCtr_att(1,:,NetId), ...
    R.cvLoss_RRR_ctrSpk_unsyncCtr_att(2,:,NetId), ...
    'o-', 'Color', clr(2,:), 'MarkerFaceColor', clr(3,:), 'MarkerSize', mksize, 'DisplayName','RRR unsync')
hold on

errorbar(0, 1-R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_att(1,NetId), ...
    R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_att(2,NetId), ...
    '^-', 'Color', clr(2,:), 'MarkerFaceColor', clr(3,:), 'MarkerSize', mksize, 'DisplayName','full model unsync')

optDim = R.optDim_ctrSpk_unsyncCtr_att(NetId);
txt_y1 = 1 + diff(R.cvLoss_RRR_ctrSpk_unsyncCtr_att(:,optDim,NetId)) + 0.01;
text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))


[ylim_min1, ylim_max1] = get_ylim(R.cvLoss_ridge_opt_ctrSpk_syncCtr_att(:,NetId), R.cvLoss_RRR_ctrSpk_syncCtr_att(:,:,NetId));

[ylim_min2, ylim_max2] = get_ylim(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_att(:,NetId), R.cvLoss_RRR_ctrSpk_unsyncCtr_att(:,:,NetId));


ylim(subplot1, [min(ylim_min1,ylim_min2), max(ylim_max1,ylim_max2)])
xlim([-1, max(x_plt)+1])

xlabel('number of predicitve dimensions')
ylabel('prediction performance')
title('sens > asso; centre group; att')

legend('location', 'best')

sgtitle(['#',num2str(NetId)])
%
if ridgeInit
    fig_file_name = [condi,'_ridgeRRR', saveFigSuffix, '_', num2str(NetId),  '.jpg'];
else
    fig_file_name = [condi,'_RRR', saveFigSuffix, '_', num2str(NetId), '.jpg'];
end

legend('location', 'best')
%
if save_fig
saveas(fig1, [results_dir, fig_file_name]);
close(fig1)
end

end
%% average across realizations

%% compare number of prediction dimensions and performance between attention conditions

%%
% endind = 20;
selectind = 1:NetNum;

fig = figure;
fig.Position = [300,300,900,400];

% number of predictive dimensions
subplot(1,2,1)

optDim_all_noatt = R.optDim_ctrSpk_syncCtr_noatt(selectind);
m_noatt = mean(optDim_all_noatt);
sem_noatt = std(optDim_all_noatt)/sqrt(length(optDim_all_noatt));
bar(1, m_noatt, 'FaceColor', clr(1,:))
hold on
errorbar(1, m_noatt, sem_noatt , 'Color',[0,0,0])

optDim_all_att = R.optDim_ctrSpk_syncCtr_att(selectind);
m_att = mean(optDim_all_att);
sem_att = std(optDim_all_att)/sqrt(length(optDim_all_att));

bar(2, m_att, 'FaceColor', clr(2,:))

errorbar(2, m_att,sem_att, 'Color',[0,0,0])

xticks([ 1 2 ])
xticklabels({'no-att','att'})
y_text = max(m_noatt+sem_noatt,m_att+sem_att)*1.2;
text(1.5, y_text, sprintf('p = %.5f', ttest(optDim_all_noatt,optDim_all_att))) % ranksum
plot([1 1 2 2], [y_text-0.4, y_text-0.2, y_text-0.2, y_text-0.4], 'Color', [0,0,0])

hold off

xlim([0,3])
ylim([0, y_text*1.1])
title('number of predictive dimensions')
%legend('noatt','att','location','best')
ylabel('number of predictive dimensions')

%
% prediction performance
subplot(1,2,2)
% RRR noatt
perf_opt_RRR_noatt = 1 - R.cvLoss_RRR_opt_ctrSpk_syncCtr_noatt(1,selectind);
m_noatt = mean(perf_opt_RRR_noatt);
sem_noatt = std(perf_opt_RRR_noatt)/sqrt(length(perf_opt_RRR_noatt));
f1 = bar(1, m_noatt, 'FaceColor', clr(1,:));
hold on
errorbar(1, m_noatt, sem_noatt , 'Color',[0,0,0])

% RRR att
perf_opt_RRR_att = 1 - R.cvLoss_RRR_opt_ctrSpk_syncCtr_att(1,selectind);
m_att = mean(perf_opt_RRR_att);
sem_att = std(perf_opt_RRR_att)/sqrt(length(perf_opt_RRR_att));
f2 = bar(2, m_att, 'FaceColor', clr(2,:));
errorbar(2, m_att, sem_att , 'Color',[0,0,0])

y_text_RRR = max(m_noatt+sem_noatt,m_att+sem_att)*1.2;
[~,p] = ttest(perf_opt_RRR_noatt,perf_opt_RRR_att);

text(1., y_text_RRR, sprintf('p = %.5f', p))
plot([1 1 2 2], [y_text_RRR-0.01, y_text_RRR-0.005, y_text_RRR-0.005, y_text_RRR-0.01], 'Color', [0,0,0])

%
% ridge noatt
perf_opt_ridge_noatt = 1 - R.cvLoss_ridge_opt_ctrSpk_syncCtr_noatt(1,selectind);
m_noatt = mean(perf_opt_ridge_noatt);
sem_noatt = std(perf_opt_ridge_noatt)/sqrt(length(perf_opt_ridge_noatt));
bar(3, m_noatt, 'FaceColor', clr(1,:))
hold on
errorbar(3, m_noatt, sem_noatt , 'Color',[0,0,0])

% ridge att
perf_opt_ridge_att = 1 - R.cvLoss_ridge_opt_ctrSpk_syncCtr_att(1,selectind);
m_att = mean(perf_opt_ridge_att);
sem_att = std(perf_opt_ridge_att)/sqrt(length(perf_opt_ridge_att));
bar(4, m_att, 'FaceColor', clr(2,:))
errorbar(4, m_att, sem_att , 'Color',[0,0,0])

y_text_ridge = max(m_noatt+sem_noatt,m_att+sem_att)*1.2;
[~,p] = ttest(perf_opt_ridge_noatt,perf_opt_ridge_att);
text(3., y_text_ridge, sprintf('p = %.5f', p))
plot([3 3 4 4], [y_text_ridge-0.01, y_text_ridge-0.005, y_text_ridge-0.005, y_text_ridge-0.01], 'Color', [0,0,0])

y_text_across = max(y_text_RRR, y_text_ridge)*1.2;

[~,p] = ttest(perf_opt_RRR_noatt,perf_opt_ridge_noatt);
text(2., y_text_across, sprintf('p = %.5f', p))

plot([1 1 3 3], [y_text_across-0.01, y_text_across-0.005, y_text_across-0.005, y_text_across-0.01], 'Color', [0,0,0])

[~,p] = ttest(perf_opt_RRR_att,perf_opt_ridge_att);
text(3., y_text_across*1.15, sprintf('p = %.5f', p))
plot([2 2 4 4], [y_text_across*1.15-0.01, y_text_across*1.15-0.005, y_text_across*1.15-0.005, y_text_across*1.15-0.01], 'Color', [0,0,0])


legend([f1, f2], 'no-att', 'att')

xticks([ 1.5 3.5])
xticklabels({'RRR','full model'})
%
ylim([0, y_text_across*1.5])

title('prediction performance')
ylabel('prediction performance')


if ridgeInit
    sgtitle([sprintf('average across %d Network realizations; ', NetNum), 'ridge RRR'])
else
    sgtitle([sprintf('average across %d Network realizations; ', NetNum), 'RRR'])
end

%
if ridgeInit
    fig_file_name = [condi,'_ridgeRRR', saveFigSuffix, '_averageDimPerf',  '.jpg'];
else
    fig_file_name = [condi,'_RRR', saveFigSuffix, '_averageDimPerf', '.jpg'];
end

if save_fig
saveas(fig, [results_dir, fig_file_name]);
close(fig)
end

%% number of prediction dimensions versus performance; average 

fig2 = figure;
fig2.Position = [200,200,800,500];

%%%     ctrSpk_syncCtr noatt
subplot1 = subplot(1,1,1);
x_plt = numDimsUsedForPrediction;

m_rrr = mean(R.cvLoss_RRR_ctrSpk_syncCtr_noatt(1,:,:), 3);
sem_rrr = std(R.cvLoss_RRR_ctrSpk_syncCtr_noatt(1,:,:), 0, 3)/sqrt(NetNum);

errorbar(x_plt, 1-m_rrr, ...
    sem_rrr, ...
    'o-', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','RRR sync no-att')
hold on

m_full = mean(R.cvLoss_ridge_opt_ctrSpk_syncCtr_noatt(1,:), 2);
sem_full = std(R.cvLoss_ridge_opt_ctrSpk_syncCtr_noatt(1,:), 0, 2)/sqrt(NetNum);

errorbar(0, 1-m_full, ...
    sem_full, ...
    '^-', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','full model sync no-att')

% % optDim = get_optDim(m_rrr, sem_rrr);
% % txt_y1 = 1 - m_rrr(optDim) + sem_rrr(optDim) + 0.01;
% % text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))

[ylim_min1_noatt, ylim_max1_noatt] = get_ylim([m_full;sem_full], [m_rrr;sem_rrr]);

%
%%%     ctrSpk_unsyncCtr noatt
m_rrr = mean(R.cvLoss_RRR_ctrSpk_unsyncCtr_noatt(1,:,:), 3);
sem_rrr = std(R.cvLoss_RRR_ctrSpk_unsyncCtr_noatt(1,:,:), 0, 3)/sqrt(NetNum);

errorbar(x_plt, 1-m_rrr, ...
    sem_rrr, ...
    'o-', 'Color', clr(1,:), 'MarkerFaceColor', clr(3,:), 'MarkerSize', mksize, 'DisplayName','RRR unsync no-att')
hold on

m_full = mean(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_noatt(1,:), 2);
sem_full = std(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_noatt(1,:), 0, 2)/sqrt(NetNum);

errorbar(0, 1-m_full, ...
    sem_full, ...
    '^-', 'Color', clr(1,:), 'MarkerFaceColor', clr(3,:), 'MarkerSize', mksize, 'DisplayName','full model unsync no-att')

% % optDim = get_optDim(m_rrr, sem_rrr);
% % txt_y1 = 1 - m_rrr(optDim) + sem_rrr(optDim) + 0.01;
% % text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))


[ylim_min2_noatt, ylim_max2_noatt] = get_ylim([m_full;sem_full], [m_rrr;sem_rrr]);


% % ylim(subplot1, [min(ylim_min1,ylim_min2), max(ylim_max1,ylim_max2)])
% % xlim([-1, max(x_plt)+1])

% xlabel('number of predicitve dimensions')
% ylabel('prediction performance')
% title('sens > asso; centre group; noatt')
% 
% legend('location', 'best')


%%%     ctrSpk_syncCtr att
% subplot1 = subplot(1,2,2);
x_plt = numDimsUsedForPrediction;

m_rrr = mean(R.cvLoss_RRR_ctrSpk_syncCtr_att(1,:,:), 3);
sem_rrr = std(R.cvLoss_RRR_ctrSpk_syncCtr_att(1,:,:), 0, 3)/sqrt(NetNum);

errorbar(x_plt, 1-m_rrr, ...
    sem_rrr, ...
    'o-', 'Color', clr(2,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','RRR sync att')
hold on

m_full = mean(R.cvLoss_ridge_opt_ctrSpk_syncCtr_att(1,:), 2);
sem_full = std(R.cvLoss_ridge_opt_ctrSpk_syncCtr_att(1,:), 0, 2)/sqrt(NetNum);

errorbar(0, 1-m_full, ...
    sem_full, ...
    '^-', 'Color', clr(2,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','full model sync att')

% % optDim = get_optDim(m_rrr, sem_rrr);
% % txt_y1 = 1 - m_rrr(optDim) + sem_rrr(optDim) + 0.01;
% % text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))

[ylim_min1_att, ylim_max1_att] = get_ylim([m_full;sem_full], [m_rrr;sem_rrr]);

%
%%%     ctrSpk_unsyncCtr att
m_rrr = mean(R.cvLoss_RRR_ctrSpk_unsyncCtr_att(1,:,:), 3);
sem_rrr = std(R.cvLoss_RRR_ctrSpk_unsyncCtr_att(1,:,:), 0, 3)/sqrt(NetNum);

errorbar(x_plt, 1-m_rrr, ...
    sem_rrr, ...
    'o-', 'Color', clr(2,:), 'MarkerFaceColor', clr(3,:), 'MarkerSize', mksize, 'DisplayName','RRR unsync att')
hold on

m_full = mean(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_att(1,:), 2);
sem_full = std(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_att(1,:), 0, 2)/sqrt(NetNum);

errorbar(0, 1-m_full, ...
    sem_full, ...
    '^-', 'Color', clr(2,:), 'MarkerFaceColor', clr(3,:), 'MarkerSize', mksize, 'DisplayName','full model unsync att')

% % optDim = get_optDim(m_rrr, sem_rrr);
% % txt_y1 = 1 - m_rrr(optDim) + sem_rrr(optDim) + 0.01;
% % text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))


[ylim_min2_att, ylim_max2_att] = get_ylim([m_full;sem_full], [m_rrr;sem_rrr]);


ylim(subplot1, [min(min(ylim_min1_noatt,ylim_min2_noatt), min(ylim_min1_att,ylim_min2_att))-0.01, ...
    max(max(ylim_max1_noatt,ylim_max2_noatt), max(ylim_max1_att,ylim_max2_att))+0.005])
xlim([-1, max(x_plt)+7])

xlabel('number of predicitve dimensions')
ylabel('prediction performance')
if topdown
    title('asso > sens; centre group; stim') % title('sens > asso; centre group; att')
else
    title('sens > asso; centre group; stim') % title('sens > asso; centre group; att')
end    
legend('location', 'northeast')

sgtitle(sprintf('average across %d Network realizations; ', NetNum))

% average number of dimensions and performance
% noatt
m_noatt_rrr = mean(perf_opt_RRR_noatt);
sem_noatt_rrr = std(perf_opt_RRR_noatt)/sqrt(length(perf_opt_RRR_noatt));

m_Dim_noatt_rrr = mean(optDim_all_noatt);
sem_Dim_noatt_rrr = std(optDim_all_noatt)/sqrt(length(optDim_all_noatt));

errorbar(m_Dim_noatt_rrr, m_noatt_rrr, sem_noatt_rrr, sem_noatt_rrr,sem_Dim_noatt_rrr,sem_Dim_noatt_rrr, ...
    '-s', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','average')

txt_y1 = m_noatt_rrr + sem_noatt_rrr + 0.01;
text(m_Dim_noatt_rrr-0.2, txt_y1, append('\downarrow','Dim:',num2str(m_Dim_noatt_rrr)))
% att
m_att_rrr = mean(perf_opt_RRR_att);
sem_att_rrr = std(perf_opt_RRR_att)/sqrt(length(perf_opt_RRR_att));

m_Dim_att_rrr = mean(optDim_all_att);
sem_Dim_att_rrr = std(optDim_all_att)/sqrt(length(optDim_all_att));

errorbar(m_Dim_att_rrr, m_att_rrr, sem_att_rrr, sem_att_rrr,sem_Dim_att_rrr,sem_Dim_att_rrr, ...
    '-s', 'Color', clr(2,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','average')

txt_y1 = m_att_rrr + sem_att_rrr + 0.01;
text(m_Dim_att_rrr-0.2, txt_y1, append('\downarrow','Dim:',num2str(m_Dim_att_rrr)))

%

if ridgeInit
    fig_file_name = [condi,'_ridgeRRR', saveFigSuffix, '_average',  '.jpg'];
else
    fig_file_name = [condi,'_RRR', saveFigSuffix, '_average', '.jpg'];
end

if save_fig
saveas(fig2, [results_dir, fig_file_name]);
close(fig2)
end


end

%%

function d_cell = sub_mean(d_cell)
    for ii = 1:numel(d_cell)
        d_cell{ii} = double(d_cell{ii}) - mean(d_cell{ii}, 1);
    end    
end
    
function [optDim] = get_optDim(m_rrr, sem_rrr)
[~,marg] = max(1-m_rrr);
optDim = find(1 - m_rrr > 1 - m_rrr(marg) - sem_rrr(marg));
optDim = optDim(1);
end

function [ylim_min, ylim_max] = get_ylim(cvLoss_ridge_opt, cvLoss_RRR)

ylim_min = min(1-cvLoss_ridge_opt(1) - cvLoss_ridge_opt(2) - 0.01, ...
min(1 - cvLoss_RRR(1,:) - cvLoss_RRR(2,:)) - 0.01);
ylim_max = max(1-cvLoss_ridge_opt(1) + cvLoss_ridge_opt(2) + 0.03, ...
max(1 - cvLoss_RRR(1,:) + cvLoss_RRR(2,:)) + 0.03);

end

function [lambdaOpt, cvLoss_select_ridge, cvLoss_ridge] = cv_RidgeReg(X, Y, C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale)
% Cross-validate Ridge Regression
% C_RIDGE_D_MAX_SHRINKAGE_FACTOR = .5:.01:1;
% 

lambda = GetRidgeLambda(C_RIDGE_D_MAX_SHRINKAGE_FACTOR, X, ...
    'Scale', scale);
[lambdaOpt, cvLoss_ridge] = RegressModelSelect(@RidgeRegress, Y, X, lambda, ...
    'Scale', scale);

cvLoss_select_ridge = cvLoss_ridge(:, lambda==lambdaOpt);

end



function [lambdaOpt_net, cvLoss_ridge_opt_net, cvLoss_ridge_net] = cv_RidgeReg_acrossNet(src_cell, trg_cell, C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale)
% Cross-validate Ridge Regression
% C_RIDGE_D_MAX_SHRINKAGE_FACTOR = .5:.01:1;
% 

net_num = length(src_cell);
lambdaOpt_net = zeros(1, net_num);
cvLoss_ridge_opt_net =  zeros(2, net_num);
cvLoss_ridge_net = zeros(2, length(C_RIDGE_D_MAX_SHRINKAGE_FACTOR), net_num);


for net = 1:net_num
    
    lambda = GetRidgeLambda(C_RIDGE_D_MAX_SHRINKAGE_FACTOR, src_cell{net}, ...
        'Scale', scale);
    [lambdaOpt, cvLoss_ridge] = RegressModelSelect(@RidgeRegress, trg_cell{net}, src_cell{net}, lambda, ...
        'Scale', scale, 'LossMeasure', 'NSE');  % NSE or MSE

    lambdaOpt_net(net) = lambdaOpt;
    cvLoss_ridge_net(:,:,net) = cvLoss_ridge;
    cvLoss_ridge_opt_net(:, net) = cvLoss_ridge(:, lambda==lambdaOpt);

end
end

function [optDim, cvLoss_RRR_opt, cvLoss_RRR] = cv_RRR_acrossNet(src_cell, trg_cell, numDimsUsedForPrediction, ridgeInit, scale)

cvNumFolds = 10;

cvOptions = statset('crossval');

regressMethod = @ReducedRankRegress;


net_num = length(src_cell); % number of random realizations of network 
optDim = zeros(1, net_num);
cvLoss_RRR_opt = zeros(2, net_num);
cvLoss_RRR = zeros(2, length(numDimsUsedForPrediction), net_num);
for net = 1:net_num

    [cvLoss_RRR_, optDimReducedRankRegress] = cv_RRR(src_cell{net}, trg_cell{net}, ...
        numDimsUsedForPrediction, cvNumFolds, cvOptions, regressMethod, ridgeInit, scale);

    optDim(net) = optDimReducedRankRegress;
    cvLoss_RRR_opt(:,net) = cvLoss_RRR_(:, numDimsUsedForPrediction==optDimReducedRankRegress);
    cvLoss_RRR(:,:,net) = cvLoss_RRR_;

end


end


function [cvLoss_RRR, optDimReducedRankRegress] = cv_RRR(X, Y, numDimsUsedForPrediction, cvNumFolds, cvOptions, regressMethod, ...
    ridgeInit, scale)

cvFun = @(Ytrain, Xtrain, Ytest, Xtest) RegressFitAndPredict...
	(regressMethod, Ytrain, Xtrain, Ytest, Xtest, ...
	numDimsUsedForPrediction, 'LossMeasure', 'NSE', ... % NSE or MSE
    'RidgeInit', ridgeInit, 'Scale', scale);

cvl = crossval(cvFun, Y, X, ...
	  'KFold', cvNumFolds, ...
	'Options', cvOptions);

% Store cross-validation results: mean loss and standard error of the
% mean across folds.
cvLoss_RRR = [ mean(cvl); std(cvl)/sqrt(cvNumFolds) ];

% To compute the optimal dimensionality for the regression model, call
% ModelSelect:
optDimReducedRankRegress = ModelSelect...
	(cvLoss_RRR, numDimsUsedForPrediction);

end


%end
