import os
import sys
import numpy as np
import pandas as pd
import pingouin as pg

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

for idx,sub in enumerate(subjectsCtrl):
    h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_ctrl, sub=sub)
    params  =  pd.read_hdf(h5_file, 'data')
    params['sub'] = np.ones(params.shape[0])*int(subjectsCtrl_newName[idx][-2:])
    paramsAll = pd.concat([paramsAll, params])

paramsAll.reset_index(inplace=True)
idx = paramsAll['condition'].isin(cond100)
paramsAll_100 = paramsAll[idx].copy()
paramsAll_100.reset_index(inplace=True)

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
    
    model = LinearRegression(fit_intercept=True)
    model.fit(np.asarray(x_train).reshape(-1, 1), y_train)
    
    # eyeSpeed_test = model.coef_*targetSpeed_test + model.intercept_
    targetSpeed_pred = (eyeSpeed_test - model.intercept_)/model.coef_
    
    results = dict()
    results['sub']          = sub
    results['intercept']    = model.intercept_
    results['coefficients'] = model.coef_
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
            tau = ((meanTgSpeed_test-11)/22)/2
        else:
            tau = ((meanTgSpeed_test-33)/-22)/2
        mean_tw_integration = mean_tw_integration.append(pd.DataFrame([[sub,cond,closestTw, tau, meanTgSpeed_test]], columns=keysMTWI))

mean_tw_integration['integration_tw'] = mean_tw_integration['integration_tw'].astype(float)

keys2plot = ['sub', 'condition', 'meanAntiVel', 'meanPredTgVel']
data2plot = pd.DataFrame([], columns=keys2plot)
for index, row in paramsFit.iterrows():
    for tg_speed in np.unique(row['targetSpeed_train']):
        idx = np.where(row['targetSpeed_train']==tg_speed)[0].astype(int)
        mean = np.mean(np.array(row['eyeSpeed_train'])[idx]) 
        meanTg = float(tg_speed)
        data2plot = data2plot.append(pd.DataFrame([[row['sub'], tg_speed, mean, meanTg]], columns=keys2plot))
    
    for tg_speed in np.unique(row['targetSpeed_test']):
        idx = np.where(row['targetSpeed_test']==tg_speed)[0]
        mean = np.mean(np.array(row['eyeSpeed_test'])[idx])
        meanTg = np.mean(np.array(row['targetSpeed_pred'])[idx])
        data2plot = data2plot.append(pd.DataFrame([[row['sub'], tg_speed, mean, meanTg]], columns=keys2plot))
        
data2plot.reset_index(inplace=True)

cmap_blues = plt.get_cmap('Blues_r') 
cmap_reds = plt.get_cmap('Reds_r') 

blues2 = cmap_blues(np.linspace(0.2, .6, 3))
reds2  = cmap_reds(np.linspace(0.2, .6, 2))
colors100 = {
    11: blues2[0],
    22: blues2[1],
    33: blues2[2],
    13: reds2[0],
    31: reds2[1],
}
pf = paramsFit[~paramsFit['sub'].isin(['sub-08'])]
coef = pf.coefficients.mean()[0]
intercept = pf.intercept.mean()
x = np.arange(-15,60,1)
reg = intercept + coef*x

data2plot = data2plot[~data2plot['sub'].isin(['sub-08'])]
mean_tw_integration = mean_tw_integration[~mean_tw_integration['subject'].isin(['sub-08'])]

dt_mean = data2plot.groupby(['condition']).mean().reset_index()
dt_std  =  data2plot.groupby(['condition']).std().reset_index()
dt_mean = dt_mean[dt_mean['condition'].isin([13,31])]
dt_std  = dt_std[dt_std['condition'].isin([13,31])]

fig = plt.figure(figsize=(two_col, 7*cm))
ax = fig.add_gridspec(1, 10)
ax1 = fig.add_subplot(ax[0, 0:4])
ax1.set_title('Target speed vs aSPv')
sns.scatterplot(data=data2plot, x='meanPredTgVel',y='meanAntiVel',hue='condition', palette=colors100, s=50, ax=ax1)
plt.plot(x,reg,c='royalblue')
plt.errorbar(x=dt_mean['meanPredTgVel'],y=dt_mean['meanAntiVel'],
                        xerr=dt_std['meanPredTgVel'],yerr=dt_std['meanAntiVel'],
                        ecolor=reds2,elinewidth=3,fmt='none')
plt.xlim([-25,70])
plt.ylim([-1,10])
plt.ylabel('Mean aSPv (°/s)')
plt.xlabel('True or estimated Target Speed (°/s)')
plt.legend([ '','v11', 'vacc', 'v22', 'vdec','v33'])


ax2 = fig.add_subplot(ax[0, 4:7])
dt = data2plot[data2plot['sub']=='sub-13'].copy()
print(dt)
coef = list(paramsFit.loc[paramsFit['sub']=='sub-13','coefficients'])[0]
intercept = list(paramsFit.loc[paramsFit['sub']=='sub-13','intercept'])[0]
x = np.arange(-15,60,1)
reg = intercept + coef*x
plt.vlines(x=dt.loc[dt.condition==13,'meanPredTgVel'], ymin=-30, ymax=dt.loc[dt.condition==13,'meanAntiVel'], linestyles='dashed', color=colors100[13])
plt.hlines(y=dt.loc[dt.condition==13,'meanAntiVel'], xmin=-30, xmax=dt.loc[dt.condition==13,'meanPredTgVel'], linestyles='dashed', color=colors100[13])
plt.vlines(x=dt.loc[dt.condition==31,'meanPredTgVel'], ymin=-30, ymax=dt.loc[dt.condition==31,'meanAntiVel'], linestyles='dashed', color=colors100[31])
plt.hlines(y=dt.loc[dt.condition==31,'meanAntiVel'], xmin=-30, xmax=dt.loc[dt.condition==31,'meanPredTgVel'], linestyles='dashed', color=colors100[31])
plt.plot(x,reg,c='royalblue')
sns.scatterplot(data=dt, x='meanPredTgVel',y='meanAntiVel',hue='condition', palette=colors100, s=50, ax=ax2)
plt.xlim([-25,70])
plt.ylim([-1,8])
plt.ylabel('Mean aSPv (°/s)')
plt.xlabel('Target Speed (°/s)')
ax2.get_legend().set_visible(False)

dt = mean_tw_integration.copy()
dt.reset_index(inplace=True)
dt['cond'] = [13 if 'Va' in x.condition else 31 for idx,x in dt.iterrows()]
jitter = 0.5 * np.random.randn(len(dt))
dt['cond2'] = dt.cond + jitter
ax3 = fig.add_subplot(ax[0, 7:10])
ax3.set_title('TWI')
sns.histplot(data=dt, x='tau', hue='cond',palette=colors100, kde=True)
plt.xlabel('Mean TWI (s)')
plt.ylabel('Frequency')
plt.legend(['vacc', 'vdec'])

plt.tight_layout()

plt.savefig('{}/exp2_twi_dist.pdf'.format(output_folder))         
plt.savefig('{}/exp2_twi_dist.png'.format(output_folder))         

print('Va, tau - ttest against 0: \n', pg.ttest(dt.loc[dt.condition.isin(['Va-100_V0-0']),'tau'], 0, alternative='greater').round(4))
print('Vd, tau - ttest against 0: \n', pg.ttest(dt.loc[dt.condition.isin(['Vd-100_V0-0']),'tau'], 0, alternative='greater').round(4))



