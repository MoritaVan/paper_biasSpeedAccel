import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('./')
from functions.utils import *
from functions.updateRC import fontsizeDict,rcConfigDict

plt.rcParams.update(fontsizeDict(small=10,medium=12))
plt.rcParams.update(rcConfigDict(filepath = "./rcparams_config.json"))

cm = 1/2.54  # centimeters in inches
single_col = 8.9*cm
oneDot5_col = 12.7*cm
two_col = 18.2*cm

main_dir = "../data/"
os.chdir(main_dir)

output_folder = "{}/outputs/exp2".format(main_dir)

data_dir_exp  = "{}/biasAcceleration".format(main_dir)
data_dir_ctrl = "{}/biasAccelerationControl".format(main_dir)


subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 
              'sub-06',
                'sub-07', 'sub-08', 'sub-09', 
              'sub-10', 'sub-11', 'sub-12', 'sub-13'] # 5, 8, 9 with inverted slope

subjectsCtrl = [
                'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'
]
subjectsCtrl_newName = [
                'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18',
]

cond100 =   [
                'V1-100_V0-0', 
                'V2-100_V0-0',
                'V3-100_V0-0',
                'Va-100_V0-0', 
                'Vd-100_V0-0',
                ]


    
# print('Plotting parameters')
paramsAll = pd.DataFrame([])
for sub in subjects:
    h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_exp, sub=sub)
    params  =  pd.read_hdf(h5_file, 'data')
    params['sub'] = np.ones(len(params)) * int(sub[-2:])
    paramsAll = pd.concat([paramsAll, params])
    print(params.shape,paramsAll.shape)

for idx,sub in enumerate(subjectsCtrl):
    h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_ctrl, sub=sub)
    params  =  pd.read_hdf(h5_file, 'data')
    params['sub'] = np.ones(params.shape[0])*int(subjectsCtrl_newName[idx][-2:])
    paramsAll = pd.concat([paramsAll, params])
    print(params.shape,paramsAll.shape)

paramsAll.reset_index(inplace=True)
idx = paramsAll['condition'].isin(cond100)
paramsAll_100 = paramsAll[idx].copy()
paramsAll_100.reset_index(inplace=True)
print(paramsAll_100.sample(10))
print(paramsAll_100.shape)

# model: perceived velocity for accelerating conditions based on constant conditions

keys2keep = [
    'sub', 'condition', 'trial', 'aSPv_x', 'aSPv_y',
]

keysFit = ['sub', 'intercept', 'coefficients', #'r_sq',
          'targetSpeed_test', 'eyeSpeed_test', 'targetSpeed_pred', 
          'targetSpeed_train', 'eyeSpeed_train',
          'mean_pred_a', 'mean_pred_d']

paramsFit = pd.DataFrame([], columns=keysFit)
paramsFit = paramsFit.astype(object)

# update subjects list (exp + control)
subjects = list(set(subjects).union(subjectsCtrl_newName))
subjects.sort()

for sub in subjects:
    print(sub)
    data_100 = paramsAll_100.loc[paramsAll_100['sub']==int(sub[-2:]),keys2keep]   
    
    idx_training = [idx for idx,x in data_100.iterrows() if x.condition in ['V1-100_V0-0', 'V2-100_V0-0', 'V3-100_V0-0']]
    idx_testing  = [idx for idx,x in data_100.iterrows() if x.condition in ['Va-100_V0-0', 'Vd-100_V0-0']]
    
    trials_test = data_100.loc[idx_testing,'trial']
    
    # training: target velocity as regressor
    y_train = data_100.loc[idx_training] # eye velocity
    x_train = y_train.condition # target velocity
    
    y_train = np.sqrt((y_train['aSPv_x']**2 + y_train['aSPv_y']**2)/2)    
    x_train = [int(x[1])*11 for x in x_train]

    # testing: inverse operation, to find the "empirical" target velocity
    eyeSpeed_test     = data_100.loc[idx_testing] 
    targetSpeed_test  = eyeSpeed_test.condition

    eyeSpeed_test  = np.sqrt((eyeSpeed_test['aSPv_x']**2 + eyeSpeed_test['aSPv_y']**2)/2)
    targetSpeed_test  = [x[1] for x in targetSpeed_test]
    targetSpeed_test  = [13 if x=='a' else 31 for x in targetSpeed_test]
    
    # model = LinearRegression(fit_intercept=False)
    model = LinearRegression(fit_intercept=True)
    model.fit(np.asarray(x_train).reshape(-1, 1), y_train)
    
    # eyeSpeed_test = model.coef_*targetSpeed_test + model.intercept_
    targetSpeed_pred = (eyeSpeed_test - model.intercept_)/model.coef_
    
    results = dict()
    results['sub']          = sub
    results['intercept']    = model.intercept_
    results['coefficients'] = model.coef_
#     results['r_sq']         = model.score(np.asarray(x_test).reshape(-1, 1), y_test)  
    results['trials_test']      = trials_test
    results['targetSpeed_test'] = list(targetSpeed_test)
    results['eyeSpeed_test']    = eyeSpeed_test
    results['targetSpeed_pred'] = targetSpeed_pred
    results['targetSpeed_train'] = x_train
    results['eyeSpeed_train']    = list(y_train)
    results['mean_pred_a']  = np.mean(targetSpeed_pred[np.array(targetSpeed_test)==13])
    results['mean_pred_d']  = np.mean(targetSpeed_pred[np.array(targetSpeed_test)==31])
    results['explained_variance'] = explained_variance_score(targetSpeed_test, targetSpeed_pred)
    results['r2_score']           = r2_score(targetSpeed_test, targetSpeed_pred)
    
    paramsFit = paramsFit.append(results, ignore_index=True)

def closest(lst, K): 
    import numpy as np

    lst = np.asarray(lst) 
    idx = (np.abs(lst - K)).argmin() 
    return idx, lst[idx] 
    
timeWindows = np.arange(50,501,50)

ta = np.linspace(0,0.87,870)
va = 11 + 22*ta
td = np.linspace(0,0.72,720)
vd = 33 - 22*td

keys = ['condition', 'time_window_size', 'speed']
tw_data = pd.DataFrame([], columns=keys)
for cond in ['Va-100_V0-0','Vd-100_V0-0']:
    for t in timeWindows:
        if cond=='Va-100_V0-0':
            idx = ta<t/1000
            speed = np.mean(va[idx])
        else:
            idx = td<t/1000
            speed = np.mean(vd[idx])

        tw_data = tw_data.append(pd.DataFrame([[cond, t, speed]], columns=keys))
        
mapCond = {
    'Va-100_V0-0': 13,
    'Vd-100_V0-0': 31
}

keysTWI  = ['subject', 'condition', 'trial', 'integration_tw', 'tau', 'pred_targetSpeed']
keysMTWI = ['subject', 'condition', 'integration_tw', 'tau', 'mean_pred_targetSpeed']
tw_integration = pd.DataFrame([], columns=keysTWI)
mean_tw_integration = pd.DataFrame([], columns=keysMTWI)
for sub in subjects:
    print('Subject:',sub)

    paramsSub = paramsFit[paramsFit['sub']==sub]

    for cond in ['Va-100_V0-0','Vd-100_V0-0']:
        idxFitCond = np.where(np.array(paramsSub['targetSpeed_test'].array[0])==mapCond[cond])[0]
        targetSpeed_pred = paramsSub['targetSpeed_pred'].array[0].copy()
        targetSpeed_pred = np.array(targetSpeed_pred)[idxFitCond]
        trials = paramsSub['trials_test'].array[0].copy()
        trials = trials.iloc[idxFitCond]

        targetSpeed_pred = pd.DataFrame(targetSpeed_pred, columns=['targetSpeed_pred'], index=np.array(trials))

        twCond = tw_data[tw_data['condition']==cond].copy()
        twCond.reset_index(inplace=True)
        for trial in targetSpeed_pred.index:
            idx, closestSpeed = closest(twCond['speed'],targetSpeed_pred.at[trial, 'targetSpeed_pred'])
            closestTw = twCond.at[idx,'time_window_size']
            if cond=='Va-100_V0-0':
                tau = (targetSpeed_pred.at[trial, 'targetSpeed_pred']-11)/22
            else:
                tau = (targetSpeed_pred.at[trial, 'targetSpeed_pred']-33)/-22
                
            tw_integration = tw_integration.append(pd.DataFrame([[sub,cond,trial,closestTw, tau, targetSpeed_pred.at[trial, 'targetSpeed_pred']]], columns=keysTWI))

        # twi based on the mean predicted target speed
        meanTgSpeed_test = np.mean(np.array(targetSpeed_pred))
        idx, closestSpeed = closest(twCond['speed'],meanTgSpeed_test)
        closestTw = twCond.at[idx,'time_window_size']
        if cond=='Va-100_V0-0':
            tau = (meanTgSpeed_test-11)/22
        else:
            tau = (meanTgSpeed_test-33)/-22
        mean_tw_integration = mean_tw_integration.append(pd.DataFrame([[sub,cond,closestTw, tau, meanTgSpeed_test]], columns=keysMTWI))

mean_tw_integration['integration_tw'] = mean_tw_integration['integration_tw'].astype(float)

# fig = plt.figure(figsize=(5*4, 5*3))
# idx=0
# for sub in subjects:
#     idx += 1
#     data2plot = tw_integration[tw_integration['subject']==sub]
# #     data2plot['integration_tw'] = [str(x) for x in data2plot['integration_tw']]
    
#     plt.subplot(4,5,idx)
#     plt.title(sub)
#     plt.ylim([0,100])
#     sns.countplot(data=data2plot, x="integration_tw", hue='condition', order=timeWindows)
    
# plt.savefig('allSubs_temporalIntegrationWindow.pdf')

# fig = plt.figure(figsize=(5*4, 5*3))
# for sub in subjects:
#     data2plot = tw_integration[tw_integration['subject']==sub].copy()
#     data2plot.reset_index(inplace=True)
    
#     plt.subplot(4,5,int(sub[-2:]))
#     plt.title(sub)
#     sns.histplot(data=data2plot, x="tau", hue='condition', kde=True)
#     # plt.xlim([-5,5])
#     plt.ylim([0,50])
# plt.savefig('allSubs_tau.pdf')

# mean_tw_integration.reset_index(inplace=True)
# fig = plt.figure(figsize=(5, 5))
# data2plot = mean_tw_integration
# plt.title('Average Temporal Integration Window')

# sns.histplot(data=data2plot, x="integration_tw", hue='condition', kde=True)
    
# plt.savefig('mean_temporalIntegrationWindow.pdf')

# fig = plt.figure(figsize=(5, 5))
# data2plot = mean_tw_integration
# plt.title('Average Tau')
# # plt.ylim([0,100])
# sns.histplot(data=data2plot, x="tau", hue='condition', kde=True)
    
# plt.savefig('mean_tau.pdf')



keys2plot = ['sub', 'condition', 'meanAntiVel', 'meanPredTgVel']
data2plot = pd.DataFrame([], columns=keys2plot)
for index, row in paramsFit.iterrows():
    for tg_speed in np.unique(row['targetSpeed_train']):
        idx = np.where(row['targetSpeed_train']==tg_speed)[0].astype(int)
        mean = np.mean(np.array(row['eyeSpeed_train'])[idx]) 
        meanTg = np.nan
#         meanTg = np.mean(np.array(row['targetSpeed_train_pred'])[idx])# pred values for cte vel???
        data2plot = data2plot.append(pd.DataFrame([[row['sub'], tg_speed, mean, meanTg]], columns=keys2plot))
    
    for tg_speed in np.unique(row['targetSpeed_test']):
        idx = np.where(row['targetSpeed_test']==tg_speed)[0]
        mean = np.mean(np.array(row['eyeSpeed_test'])[idx])
        meanTg = np.mean(np.array(row['targetSpeed_pred'])[idx])
        data2plot = data2plot.append(pd.DataFrame([[row['sub'], tg_speed, mean, meanTg]], columns=keys2plot))
        
data2plot.reset_index(inplace=True)

colors = [
    np.array([92, 224, 22])/255, # green Va
    np.array([224, 80, 45])/255, # orange Vd
    ]
fig = plt.figure(figsize=(two_col*cm, 8*cm))
ax = plt.subplot(1,2,1)
dt = data2plot[data2plot['condition'].isin([13,31])]
dt_mean = dt.groupby(['condition']).mean()
dt_std =  dt.groupby(['condition']).std()
sns.scatterplot(data=dt, x='meanPredTgVel',y='meanAntiVel',hue='condition', palette=sns.diverging_palette(20, 300, s=60, as_cmap=True), s=75, ax=ax)
dt = data2plot[data2plot['condition'].isin([11,22,33])]
dt.condition = dt.condition.astype(float)
sns.scatterplot(data=dt, x='meanPredTgVel', y='meanAntiVel', 
                hue='condition', palette=sns.dark_palette("#69d", as_cmap=True), s=75, ax=ax)
plt.errorbar(x=dt_mean['meanPredTgVel'],y=dt_mean['meanAntiVel'],
                        xerr=dt_std['meanPredTgVel'],yerr=dt_std['meanAntiVel'],
                        ecolor=colors,elinewidth=3,fmt='none')
# plt.plot(dt[''])

# g = sns.JointGrid(data=dt, x=dt['meanPredTgVel'], y=dt['meanAntiVel'], 
#                     hue='cond', palette=sns.diverging_palette(20, 300, s=60, as_cmap=True), height=9)
# g.plot_joint(sns.scatterplot, s=75, alpha=1)
# g.plot_marginals(sns.histplot, kde=True)
# dt = data2plot[data2plot['cond'].isin([11,22,33])]
# dt.cond = dt.cond.astype(float)
# g = sns.JointGrid(data=dt, x=dt['meanPredTgVel'], y=dt['meanAntiVel'], 
#                     hue='cond', palette=sns.dark_palette("#69d", as_cmap=True))
# g.plot_joint(sns.scatterplot, s=75, alpha=1)
# # sns.scatterplot(data=dt, x=dt['meanPredTgVel'], y=dt['meanAntiVel'], 
# #                     hue='cond', palette=sns.dark_palette("#69d", as_cmap=True), s=75)
# sns.regplot(data=dt, x=dt['meanPredTgVel'], y=dt['meanAntiVel'], marker='')
# plt.xlim([-35,70])
# plt.ylim([0,10])
plt.savefig('allSubs_predictedTgVel_vs_antiVel.pdf')          
# plt.savefig('allSubs_predictedTgVel_vs_antiVel.svg')          


