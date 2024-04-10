function RRR(varargin)


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

% sys_arg = str2double(varargin{1});
ii = 0;

ridgeInit = 1;
topdown = 0;


scale = false;



sprintf('ridgeInit: %d',ridgeInit)


condi = 'spon';

% topdown = 1;
if topdown
    rv = '_rv';
else
    rv = '';
end
sprintf('topdown: %d',topdown)

mkdir RRR_results
if ridgeInit == 0
    results_dir = ['results/', condi, '_RRR_syncUnsync_rg5', rv, '/']; %'results/';    
else
    results_dir = ['results/', condi, '_ridgeRRR_syncUnsync_rg5', rv, '/']; %'results/';
end

mkdir(sprintf('%s',results_dir))

if ridgeInit
    data_file_name = [condi, '_ridgeRRR_syncUnsync_sua_lc_rg5', rv, '.mat']; % stim_ridgeRRR_att_whole
else
    data_file_name = [condi, '_RRR_syncUnsync_sua_lc_rg5', rv, '.mat']; % stim_RRR_att_whole
end   

saveFigSuffix = ['_syncUnsync_lc_rg5',rv]; % _whole





%%


    
load(['raw_data/spon_rg5_ctrsua_sync_local_subM1.mat'])
load(['raw_data/spon_rg5_ctrsua_unsync_local_subM1.mat'])
% 
if topdown
    X_ctrSpk_syncCtr_spon = a2_rg5_ctrsua_sync_spon_lc_ctr;
    y_ctrSpk_syncCtr_spon = a1_rg5_ctrsua_sync_spon_lc_ctr;

    X_ctrSpk_unsyncCtr_spon = a2_rg5_ctrsua_unsync_spon_lc_ctr;
    y_ctrSpk_unsyncCtr_spon = a1_rg5_ctrsua_unsync_spon_lc_ctr;

else
    X_ctrSpk_syncCtr_spon = a1_rg5_ctrsua_sync_spon_lc_ctr;
    y_ctrSpk_syncCtr_spon = a2_rg5_ctrsua_sync_spon_lc_ctr;

    X_ctrSpk_unsyncCtr_spon = a1_rg5_ctrsua_unsync_spon_lc_ctr;
    y_ctrSpk_unsyncCtr_spon = a2_rg5_ctrsua_unsync_spon_lc_ctr;
end
    
% X_ctrSpk_syncCtr_spon = a1_rg5_ctrsua_sync_sponL_lc_ctr;
% y_ctrSpk_syncCtr_spon = a2_rg5_ctrsua_sync_sponL_lc_ctr;
% 
% X_ctrSpk_unsyncCtr_spon = a1_rg5_ctrsua_unsync_sponL_lc_ctr;
% y_ctrSpk_unsyncCtr_spon = a2_rg5_ctrsua_unsync_sponL_lc_ctr;


% % X_ctrSpk_unsyncCtr_spon = a1_rg5_ctrsua_unsyncEtr_sponL_lc_ctr;
% % y_ctrSpk_unsyncCtr_spon = a2_rg5_ctrsua_unsyncEtr_sponL_lc_ctr;

    
% X_corSpk_syncCor = a1_rg5_corsua_sync_sponL_lc_cor;
% y_corSpk_syncCor = a2_rg5_corsua_sync_sponL_lc_cor;
% 
% X_corSpk_syncCtr = a1_rg5_corsua_sync_sponL_lc_ctr;
% y_corSpk_syncCtr = a2_rg5_corsua_sync_sponL_lc_ctr;


% X_att_ctr = a1_sua_sync_att_lc_ctr;
% Y_noatt_ctr = a2_sua_sync_noatt_lc_ctr;
% Y_att_ctr = a2_sua_sync_att_lc_ctr;
% X_noatt_cor = a1_sua_sync_noatt_lc_cor;
% Y_noatt_cor = a2_sua_sync_noatt_lc_cor;



%% centre spk sync ctr noatt
src = X_ctrSpk_syncCtr_spon;
trg = y_ctrSpk_syncCtr_spon;

[lambdaOpt, cvLoss_ridge_opt, cvLoss_ridge] = cv_RidgeReg_acrossNet(src, trg, ...
    C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale);

R.lambdaOpt_ctrSpk_syncCtr_spon = lambdaOpt; 
R.cvLoss_ridge_opt_ctrSpk_syncCtr_spon = cvLoss_ridge_opt ;
R.cvLoss_ridge_ctrSpk_syncCtr_spon = cvLoss_ridge;


[optDim, cvLoss_RRR_opt, cvLoss_RRR] = cv_RRR_acrossNet(src, ...
    trg, numDimsUsedForPrediction, ridgeInit, scale);

R.optDim_ctrSpk_syncCtr_spon = optDim; 
R.cvLoss_RRR_opt_ctrSpk_syncCtr_spon = cvLoss_RRR_opt;
R.cvLoss_RRR_ctrSpk_syncCtr_spon = cvLoss_RRR;


%% centre spk unsync ctr noatt
src = X_ctrSpk_unsyncCtr_spon;
trg = y_ctrSpk_unsyncCtr_spon;

[lambdaOpt, cvLoss_ridge_opt, cvLoss_ridge] = cv_RidgeReg_acrossNet(src, trg, ...
    C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale);

R.lambdaOpt_ctrSpk_unsyncCtr_spon = lambdaOpt; 
R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_spon = cvLoss_ridge_opt ;
R.cvLoss_ridge_ctrSpk_unsyncCtr_spon = cvLoss_ridge;


[optDim, cvLoss_RRR_opt, cvLoss_RRR] = cv_RRR_acrossNet(src, ...
    trg, numDimsUsedForPrediction, ridgeInit, scale);

R.optDim_ctrSpk_unsyncCtr_spon = optDim; 
R.cvLoss_RRR_opt_ctrSpk_unsyncCtr_spon = cvLoss_RRR_opt;
R.cvLoss_RRR_ctrSpk_unsyncCtr_spon = cvLoss_RRR;


%% save data
save([results_dir, data_file_name], 'R')



%%

clr = [100 100 100; 255 255 255]/255;

mksize = 8;

NetNum = numel(X_ctrSpk_syncCtr_spon);
%%
for NetId=1:NetNum

fig1 = figure;
fig1.Position = [200,200,1200,500];

%%%     ctrSpk_syncCtr spon
subplot1 = subplot(1,2,1);
x_plt = numDimsUsedForPrediction;
errorbar(x_plt, 1-R.cvLoss_RRR_ctrSpk_syncCtr_spon(1,:,NetId), ...
    R.cvLoss_RRR_ctrSpk_syncCtr_spon(2,:,NetId), ...
    'o-', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','RRR sync')
hold on

errorbar(0, 1-R.cvLoss_ridge_opt_ctrSpk_syncCtr_spon(1,NetId), ...
    R.cvLoss_ridge_opt_ctrSpk_syncCtr_spon(2,NetId), ...
    '^-', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','full model sync')

optDim = R.optDim_ctrSpk_syncCtr_spon(NetId);
txt_y1 = 1 + diff(R.cvLoss_RRR_ctrSpk_syncCtr_spon(:,optDim,NetId)) + 0.002;
text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))

%%%     ctrSpk_unsyncCtr spon
errorbar(x_plt, 1-R.cvLoss_RRR_ctrSpk_unsyncCtr_spon(1,:,NetId), ...
    R.cvLoss_RRR_ctrSpk_unsyncCtr_spon(2,:,NetId), ...
    'o-', 'Color', clr(1,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','RRR unsync')
hold on

errorbar(0, 1-R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_spon(1,NetId), ...
    R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_spon(2,NetId), ...
    '^-', 'Color', clr(1,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','full model unsync')

optDim = R.optDim_ctrSpk_unsyncCtr_spon(NetId);
txt_y1 = 1 + diff(R.cvLoss_RRR_ctrSpk_unsyncCtr_spon(:,optDim,NetId)) + 0.002;
text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))


[ylim_min1, ylim_max1] = get_ylim(R.cvLoss_ridge_opt_ctrSpk_syncCtr_spon(:,NetId), R.cvLoss_RRR_ctrSpk_syncCtr_spon(:,:,NetId));

[ylim_min2, ylim_max2] = get_ylim(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_spon(:,NetId), R.cvLoss_RRR_ctrSpk_unsyncCtr_spon(:,:,NetId));


ylim(subplot1, [min(ylim_min1,ylim_min2), max(ylim_max1,ylim_max2)])
xlim([-1, max(x_plt)+1])

xlabel('number of predicitve dimensions')
ylabel('prediction performance')
title('sens > asso; centre group; spon')

legend('location', 'best')

sgtitle(['#',num2str(NetId)])
%
if ridgeInit
    fig_file_name = [condi,'_ridgeRRR', saveFigSuffix, '_', num2str(NetId),  '.jpg'];
else
    fig_file_name = [condi,'_RRR', saveFigSuffix, '_', num2str(NetId), '.jpg'];
end
%%
if save_fig
saveas(fig1, [results_dir, fig_file_name]);
close(fig1)
end

end
%%
%% average across realizations
%% average number of prediction dimensions and performance
fig = figure;
fig.Position = [300,300,900,400];

% number of predictive dimensions
subplot(1,2,1)

optDim_all_spon = R.optDim_ctrSpk_syncCtr_spon;
m_spon = mean(optDim_all_spon);
sem_spon = std(optDim_all_spon)/sqrt(length(optDim_all_spon));
bar(1, m_spon, 'FaceColor', clr(1,:))
hold on
errorbar(1, m_spon, sem_spon , 'Color',[0,0,0])


xticks( 1 )
xticklabels({'spon'})
% y_text = max(m_spon+sem_spon)*1.2;
% text(1.5, y_text, sprintf('p = %.5f', ranksum(optDim_all_spon,optDim_all_att)))
% plot([1 1 2 2], [y_text-0.4, y_text-0.2, y_text-0.2, y_text-0.4], 'Color', [0,0,0])

hold off

xlim([0,2])
ylim([0, (m_spon+sem_spon)*1.2])
title('number of predictive dimensions')
%legend('noatt','att','location','best')
ylabel('number of predictive dimensions')

%
% prediction performance
subplot(1,2,2)
% RRR spon
perf_opt_RRR_spon = 1 - R.cvLoss_RRR_opt_ctrSpk_syncCtr_spon(1,:);
m_spon_rrr = mean(perf_opt_RRR_spon);
sem_spon_rrr = std(perf_opt_RRR_spon)/sqrt(length(perf_opt_RRR_spon));
f1 = bar(1, m_spon_rrr, 'FaceColor', clr(1,:));
hold on
errorbar(1, m_spon_rrr, sem_spon_rrr , 'Color',[0,0,0])


% ridge spon
perf_opt_ridge_spon = 1 - R.cvLoss_ridge_opt_ctrSpk_syncCtr_spon(1,:);
m_spon_ridge = mean(perf_opt_ridge_spon);
sem_spon_ridge = std(perf_opt_ridge_spon)/sqrt(length(perf_opt_ridge_spon));
f2 = bar(2, m_spon_ridge, 'FaceColor', clr(1,:));
hold on
errorbar(2, m_spon_ridge, sem_spon_ridge , 'Color',[0,0,0])


y_text_across = max(m_spon_rrr+sem_spon_rrr,m_spon_ridge+sem_spon_ridge)*1.2;

text(1.5, y_text_across, sprintf('p = %.5f', ranksum(perf_opt_RRR_spon,perf_opt_ridge_spon)))
plot([1 1 2 2], [y_text_across*0.90, y_text_across*0.93, y_text_across*0.93, y_text_across*0.90], 'Color', [0,0,0])



xticks([ 1 2])
xticklabels({'RRR','full model'})

ylim([0, y_text_across*1.3])

title('prediction performance')
ylabel('prediction performance')


if ridgeInit
    sgtitle([sprintf('average across %d Network realizations; spon sync; ', NetNum), 'ridge RRR'])
else
    sgtitle([sprintf('average across %d Network realizations; spon sync; ', NetNum), 'RRR'])
end

%%
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

%%%     ctrSpk_syncCtr spon
subplot1 = subplot(1,1,1);
x_plt = numDimsUsedForPrediction;

m_rrr = mean(R.cvLoss_RRR_ctrSpk_syncCtr_spon(1,:,:), 3);
sem_rrr = std(R.cvLoss_RRR_ctrSpk_syncCtr_spon(1,:,:), 0, 3)/sqrt(NetNum);

errorbar(x_plt, 1-m_rrr, ...
    sem_rrr, ...
    'o-', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','RRR sync')
hold on

m_full = mean(R.cvLoss_ridge_opt_ctrSpk_syncCtr_spon(1,:), 2);
sem_full = std(R.cvLoss_ridge_opt_ctrSpk_syncCtr_spon(1,:), 0, 2)/sqrt(NetNum);

errorbar(0, 1-m_full, ...
    sem_full, ...
    '^-', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize, 'DisplayName','full model sync')

optDim = get_optDim(m_rrr, sem_rrr);
txt_y1 = 1 - m_rrr(optDim) + sem_rrr(optDim) + 0.01;
text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))

[ylim_min1_spon, ylim_max1_spon] = get_ylim([m_full;sem_full], [m_rrr;sem_rrr]);

%
%%%     ctrSpk_unsyncCtr spon
m_rrr = mean(R.cvLoss_RRR_ctrSpk_unsyncCtr_spon(1,:,:), 3);
sem_rrr = std(R.cvLoss_RRR_ctrSpk_unsyncCtr_spon(1,:,:), 0, 3)/sqrt(NetNum);

errorbar(x_plt, 1-m_rrr, ...
    sem_rrr, ...
    'o-', 'Color', clr(1,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','RRR unsync')
hold on

m_full = mean(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_spon(1,:), 2);
sem_full = std(R.cvLoss_ridge_opt_ctrSpk_unsyncCtr_spon(1,:), 0, 2)/sqrt(NetNum);

errorbar(0, 1-m_full, ...
    sem_full, ...
    '^-', 'Color', clr(1,:), 'MarkerFaceColor', clr(2,:), 'MarkerSize', mksize, 'DisplayName','full model unsync')

% optDim = get_optDim(m_rrr, sem_rrr);
% txt_y1 = 1 - m_rrr(optDim) + sem_rrr(optDim) + 0.01;
% text(optDim-0.2, txt_y1, append('\downarrow','Dim:',num2str(optDim)))


[ylim_min2_spon, ylim_max2_spon] = get_ylim([m_full;sem_full], [m_rrr;sem_rrr]);


ylim(subplot1, [min(ylim_min1_spon,ylim_min2_spon), max(ylim_max1_spon,ylim_max2_spon)])
xlim([-1, max(x_plt)+1])


% average number of dimensions and performance
perf_opt_RRR_spon = 1 - R.cvLoss_RRR_opt_ctrSpk_syncCtr_spon(1,:);
m_spon_rrr = mean(perf_opt_RRR_spon);
sem_spon_rrr = std(perf_opt_RRR_spon)/sqrt(length(perf_opt_RRR_spon));

optDim_all_spon = R.optDim_ctrSpk_syncCtr_spon;
m_Dim_spon_rrr = mean(optDim_all_spon);
sem_Dim_spon_rrr = std(optDim_all_spon)/sqrt(length(optDim_all_spon));

errorbar(m_Dim_spon_rrr, m_spon_rrr, sem_spon_rrr, sem_spon_rrr,sem_Dim_spon_rrr,sem_Dim_spon_rrr, ...
    '-s', 'Color', clr(1,:), 'MarkerFaceColor', clr(1,:), 'MarkerSize', mksize+4, 'DisplayName','average')

%optDim = get_optDim(m_rrr, sem_rrr);
txt_y1 = m_spon_rrr + sem_spon_rrr + 0.001;
text(m_Dim_spon_rrr-0.2, txt_y1, append('\downarrow','Dim:',num2str(m_Dim_spon_rrr)))


xlabel('number of predicitve dimensions')
ylabel('prediction performance')
title('sens > asso; centre group; spon')

legend('location', 'best')


sgtitle(sprintf('average across %d Network realizations', NetNum))


%%
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
function [optDim] = get_optDim(m_rrr, sem_rrr)
[~,marg] = max(1-m_rrr);
optDim = find(1 - m_rrr > 1 - m_rrr(marg) - sem_rrr(marg));
optDim = optDim(1);
end

function [ylim_min, ylim_max] = get_ylim(cvLoss_ridge_opt, cvLoss_RRR)

ylim_min = min(1-cvLoss_ridge_opt(1) - cvLoss_ridge_opt(2)-0.001, ...
min(1 - cvLoss_RRR(1,:) - cvLoss_RRR(2,:))-0.001);
ylim_max = max(1-cvLoss_ridge_opt(1) + cvLoss_ridge_opt(2) + 0.003, ...
max(1 - cvLoss_RRR(1,:) + cvLoss_RRR(2,:)) + 0.003);

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

% function cvLoss_select_rmDim = remove_activity_along_predict_dim(X_rmDim, Y_rmDim, X_pred, Y_pred, numDimsUsedForPrediction, ...
%     numDimstobeDeleted, ridgeInit, scale, C_RIDGE_D_MAX_SHRINKAGE_FACTOR)
% 
% % remove activity of X_pred along predictive dimensions of X_rmDim and
% % Y_rmDim, and then use 'after-removed' X_pred to predict Y_pred 
% 
% 
% [~, B_, ~] = ReducedRankRegress(Y_rmDim, X_rmDim, numDimsUsedForPrediction, ...
%     'UseRidgeInit', ridgeInit, 'Scale', scale); % predictive dimensions of 'sync centre att'
% 
% 
% 
% % remove activity and predict using ridge regression
% [cvLoss_select_rmDim] = remove_activity_along_predict_dim_(X_pred, Y_pred, B_, numDimstobeDeleted, C_RIDGE_D_MAX_SHRINKAGE_FACTOR);
% 
% end

% function [CvLoss_select] = remove_activity_along_predict_dim_(X, Y, B_, numDimstobeDeleted, C_RIDGE_D_MAX_SHRINKAGE_FACTOR)
% 
% scale = false;
% 
% CvLoss_select = zeros(2,numel(numDimstobeDeleted));
% sig_X = cov(X);
% %dY = size(Y_V2, 2);
% ii = 1;
% for deleteDimNum = numDimstobeDeleted
%     [u, s, v_svd] = svd(B_(:, 1:deleteDimNum )' * sig_X);
%     v_orthog = v_svd(:, deleteDimNum+1:end);
%     X_orthogProj = X*v_orthog;
% 
%     [lambdaOpt, cvLoss_select_ridge, cvLoss_ridge] = cv_RidgeReg(X_orthogProj, Y, C_RIDGE_D_MAX_SHRINKAGE_FACTOR, scale);
% 
%     
%     CvLoss_select(:,ii) = cvLoss_select_ridge;
%     ii = ii + 1;
% end
% end


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
        'Scale', scale);

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
	numDimsUsedForPrediction, 'LossMeasure', 'NSE', ...
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
