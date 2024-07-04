#%%
import os
import sys
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

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
# os.chdir(main_dir)

output_folder = "{}/outputs/exp2".format(main_dir)

data_dir_exp    = "{}/biasAcceleration".format(main_dir)
data_dir_ctrl   = "{}/biasAccelerationControl".format(main_dir)
data_dir_ctrlV2 = "{}/biasAccelerationControlV2".format(main_dir)


subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 
              'sub-06',
                'sub-07', 'sub-08', 'sub-09', 
              'sub-10', 'sub-11', 'sub-12', 'sub-13'] # 5, 8, 9 with inverted slope

subjectsCtrl = [
                'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05'
]
subjectsCtrlV2 = ['sub-01', 'sub-02']#, 'sub-03', 'sub-04', 'sub-05']

newSubCtrl = { # subName in control : corresponding subName in experiment
    'sub-01': 'sub-01', 
    'sub-02': 'sub-13', 
    'sub-03': 'sub-16', 
    'sub-04': 'sub-17', 
    'sub-05': 'sub-18'
}

newSubCtrlV2 = { # subName in controlV2 : corresponding subName in experiment
    'sub-01': 'sub-13', 
    'sub-02': 'sub-01', 
    # 'sub-03': 'sub-19', 
    # 'sub-04': 'sub-20', 
    # 'sub-05': 'sub-21'
}

cond100 =   {
                'V1-100_V0-0': 'V1c', 
                'V2-100_V0-0': 'V2c',
                'V3-100_V0-0': 'V3c',
                'Va-100_V0-0': 'V1a', 
                'Vd-100_V0-0': 'V3d',
}

cond100CtrlV2 = {
                'V1d-100_V0-0': 'V1d', 
                'V1a-100_V0-0': 'V1a', 
                'V1c-100_V0-0': 'V1c', 
                'V2d-100_V0-0': 'V2d',
                'V2a-100_V0-0': 'V2a',
                'V2c-100_V0-0': 'V2c',
                'V3d-100_V0-0': 'V3d',
                'V3a-100_V0-0': 'V3a',
                'V3c-100_V0-0': 'V3c',
}

#%%

    
# print('Plotting parameters')
paramsAll = pd.DataFrame()
for sub in subjects:
    h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_exp, sub=sub)
    params  =  pd.read_hdf(h5_file, 'data')
    params  = params[params['condition'].isin(cond100)]
    params['cond'] = [cond100[x] for x in params['condition']]
    params['sub'] = np.ones(len(params)) * int(sub[-2:])
    paramsAll = pd.concat([paramsAll, params])

for idx,sub in enumerate(subjectsCtrl):
    h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_ctrl, sub=sub)
    params  =  pd.read_hdf(h5_file, 'data')
    params  = params[params['condition'].isin(cond100)]
    params['cond'] = [cond100[x] for x in params['condition']]
    params['sub'] = np.ones(params.shape[0])*int(newSubCtrl[sub][-2:])
    paramsAll = pd.concat([paramsAll, params])

for idx,sub in enumerate(subjectsCtrlV2):
    h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_ctrlV2, sub=sub)
    params  =  pd.read_hdf(h5_file, 'data')
    params  = params[params['condition'].isin(cond100CtrlV2)]
    params['cond'] = [cond100CtrlV2[x] for x in params['condition']]
    params['sub'] = np.ones(params.shape[0])*int(newSubCtrl[sub][-2:])
    paramsAll = pd.concat([paramsAll, params])


paramsAll.loc[paramsAll['SPlat_x'] == paramsAll['aSPon_x']+1, 'aSPon_x'] = np.nan
paramsAll.loc[paramsAll['SPlat_y'] == paramsAll['aSPon_y']+1, 'aSPon_y'] = np.nan

paramsAll.reset_index(inplace=True)

#%%
# model: perceived velocity for accelerating conditions based on constant conditions

keys2keep = [
    'sub', 'cond', 'trial', 'aSPv_x', 'aSPv_y',
]

dataFit    = pd.DataFrame()
resultsFit = pd.DataFrame()
# update subjects list (exp + control)
subjects = list(set(subjects).union(newSubCtrl.values()).union(newSubCtrlV2.values()))
subjects.sort()

for sub in subjects:
    print(sub)
    data_100 = paramsAll.loc[paramsAll['sub']==int(sub[-2:]),keys2keep].reset_index()   
    
    idx_training = [idx for idx,x in data_100.iterrows() if 'c' in x.cond]
    idx_testing  = [idx for idx,x in data_100.iterrows() if 'c' not in x.cond]
    
    trials_test = data_100.loc[idx_testing,'trial']
    
    # training: target velocity as regressor
    y_train = data_100.loc[idx_training] # eye velocity
    x_train = y_train.cond # target velocity
    
    y_train = np.sqrt((y_train['aSPv_x']**2 + y_train['aSPv_y']**2)/2)    
    x_train = [int(x[1])*11 for x in x_train]

    # testing: inverse operation, to find the "empirical" target velocity
    eyeSpeed_test     = data_100.loc[idx_testing] 
    targetSpeed_test  = eyeSpeed_test.cond

    eyeSpeed_test  = np.sqrt((eyeSpeed_test['aSPv_x']**2 + eyeSpeed_test['aSPv_y']**2)/2)
    
    model = LinearRegression(fit_intercept=True)
    model.fit(np.asarray(x_train).reshape(-1, 1), y_train)
    
    # eyeSpeed_test = model.coef_*targetSpeed_test + model.intercept_
    targetSpeed_pred = (eyeSpeed_test - model.intercept_)/model.coef_
    
    results = dict()
    results['sub']          = sub
    results['intercept']    = model.intercept_
    results['coefficients'] = model.coef_
    resultsFit = resultsFit.append(results, ignore_index=True)
    
    data = pd.DataFrame()
    data['sub'] = data_100['sub']
    data['cond'] = data_100['cond']
    data.loc[idx_training,'trials_type'] = 'train'
    data.loc[idx_testing,'trials_type']  = 'test'
    data['targetSpeed']      = [int(x[1])*11 for x in data_100['cond']]
    data['eyeSpeed']         = np.sqrt((data_100['aSPv_x']**2 + data_100['aSPv_y']**2)/2) 
    data.loc[idx_training,'targetSpeed_pred'] = np.nan
    data.loc[idx_testing,'targetSpeed_pred']  = targetSpeed_pred  
    dataFit = pd.concat([dataFit,data], ignore_index=True)

#%%

def closest(lst, K): 
    import numpy as np

    lst = np.asarray(lst) 
    idx = (np.abs(lst - K)).argmin() 
    return idx, lst[idx] 
    
timeWindows = np.arange(50,501,50)

t  = np.linspace(0,0.7,700)
v1a = 11/np.sqrt(2) + 22/np.sqrt(2)*t
v1d = 11/np.sqrt(2) - 22/np.sqrt(2)*t
v2a = 22/np.sqrt(2) + 22/np.sqrt(2)*t
v2d = 22/np.sqrt(2) - 22/np.sqrt(2)*t
v3a = 33/np.sqrt(2) + 22/np.sqrt(2)*t
v3d = 33/np.sqrt(2) - 22/np.sqrt(2)*t

velocities = {
    'V1a': v1a,
    'V2a': v2a,
    'V3a': v3a,
    'V1d': v1d,
    'V2d': v2d,
    'V3d': v3d,
}

keys = ['condition', 'time_window_size', 'speed']
tw_data = pd.DataFrame([], columns=keys)
for cond in cond100CtrlV2.values(): #['Va-100_V0-0','Vd-100_V0-0']:
    if 'c' not in cond:
        timeArray = t
        vel = velocities[cond]
        for t in timeWindows:
            idx = timeArray<t/1000
            speed = np.mean(vel[idx])

            tw_data = tw_data.append(pd.DataFrame([[cond, t, speed]], columns=keys))
        
mapCond = {
    'V1d': 9, 
    'V1a': 13, 
    'V1c': 11, 
    'V2d': 20,
    'V2a': 24,
    'V2c': 22,
    'V3d': 31,
    'V3a': 35,
    'V3c': 33,
}

keysTWI  = ['subject', 'condition', 'integration_tw', 'tau', 'pred_targetSpeed']
keysMTWI = ['subject', 'condition', 'integration_tw', 'tau', 'mean_pred_targetSpeed']
tw_integration = pd.DataFrame([], columns=keysTWI)
mean_tw_integration = pd.DataFrame([], columns=keysMTWI)
for sub in subjects:
    print('Subject:',sub)

    paramsSub = resultsFit[resultsFit['sub']==sub]

    for cond in cond100CtrlV2.values(): #['Va-100_V0-0','Vd-100_V0-0']:
        if 'c' not in cond:
            dataFitCond = dataFit[(dataFit['sub']==int(sub[-2:]))&(dataFit['cond']==cond)]
            
            if '1' in cond: v0 = 11
            if '2' in cond: v0 = 22
            if '3' in cond: v0 = 33

            accel = 22 if 'a' in cond else -22

            twCond = tw_data[tw_data['condition']==cond].copy()
            twCond.reset_index(inplace=True)
            for trial,row in dataFitCond.iterrows():
                idx, closestSpeed = closest(twCond['speed'],row['targetSpeed_pred'])
                closestTw = twCond.at[idx,'time_window_size']
                tau = (row['targetSpeed_pred']-v0)/accel
                    
                tw_integration = tw_integration.append(pd.DataFrame([[sub,cond,closestTw, tau, row['targetSpeed_pred']]], columns=keysTWI))

            # twi based on the mean predicted target speed
            meanTgSpeed_test = np.mean(np.array(dataFitCond['targetSpeed_pred']))
            idx, closestSpeed = closest(twCond['speed'],meanTgSpeed_test)
            closestTw = twCond.at[idx,'time_window_size']
            tau = ((meanTgSpeed_test-v0)/accel)/2
            mean_tw_integration = mean_tw_integration.append(pd.DataFrame([[sub,cond,closestTw, tau, meanTgSpeed_test]], columns=keysMTWI))

mean_tw_integration['integration_tw'] = mean_tw_integration['integration_tw'].astype(float)

#%%
data2plot = dataFit.groupby(['sub', 'cond', 'trials_type']).mean().reset_index()
data2plot.loc[data2plot['cond']=='V1c','targetSpeed_pred'] = 11
data2plot.loc[data2plot['cond']=='V2c','targetSpeed_pred'] = 22
data2plot.loc[data2plot['cond']=='V3c','targetSpeed_pred'] = 33

# cmap_blues  = plt.get_cmap('Blues_r') 
# cmap_reds   = plt.get_cmap('Reds_r') 
# cmap_greens = plt.get_cmap('Greens_r') 

# blues2  = cmap_blues(np.linspace(0.2, .6, 3))
# reds2   = cmap_reds(np.linspace(0.2, .6, 3))
# greens2 = cmap_greens(np.linspace(0.2, .6, 3))

# colors100 = {
#     'V1d': greens2[0],
#     'V2d': greens2[1],
#     'V3d': greens2[2],
#     'V1c': blues2[0],
#     'V2c': blues2[1],
#     'V3c': blues2[2],
#     'V1a': reds2[0],
#     'V2a': reds2[1],
#     'V3a': reds2[2],
# }

cmap = sns.color_palette("flare", as_cmap=True)
colors = cmap(np.linspace(0, 1, 9))

colors100 = {
    'V1d': colors[0],
    'V1c': colors[1],
    'V1a': colors[2],
    'V2d': colors[3],
    'V2c': colors[4],
    'V2a': colors[5],
    'V3d': colors[6],
    'V3c': colors[7],
    'V3a': colors[8],
}

pf = resultsFit[~resultsFit['sub'].isin(['sub-08'])]
coef = pf.coefficients.mean()[0]
intercept = pf.intercept.mean()
x = np.arange(-15,60,1)
reg = intercept + coef*x

data2plot = data2plot[data2plot['sub']!=8]
mean_tw_integration = mean_tw_integration[~mean_tw_integration['subject'].isin(['sub-08'])]

dt_mean = data2plot.groupby(['cond']).mean().reset_index()
dt_std  = data2plot.groupby(['cond']).std().reset_index()

dt_mean.loc[dt_mean['cond']=='V1c','targetSpeed_pred'] = 11
dt_mean.loc[dt_mean['cond']=='V2c','targetSpeed_pred'] = 22
dt_mean.loc[dt_mean['cond']=='V3c','targetSpeed_pred'] = 33
dt_mean['color'] = [colors100[c] for c in dt_mean['cond']]

idxConst = [True if 'c' in cond else False for cond in data2plot['cond']]
idxAccel = [False if 'c' in cond else True for cond in data2plot['cond']]
idxConstMean = [idx for idx,row in dt_mean.iterrows() if 'c' in row['cond']]
idxAccelMean = [idx for idx,row in dt_mean.iterrows() if 'c' not in row['cond']]

fig = plt.figure(figsize=(two_col, 7*cm))
# ax = fig.add_gridspec(5, 10)
# ax1 = fig.add_subplot(ax[0:5, 0:5])
ax = fig.add_gridspec(1, 3)

# ax2 = fig.add_subplot(ax[0:2, 6:10])
ax2 = fig.add_subplot(ax[0, 0])
ax2.set_title('Individual example')
dt = data2plot[data2plot['sub']==13].copy()
dt.loc[dt['cond']=='V1c','targetSpeed_pred'] = 11
dt.loc[dt['cond']=='V2c','targetSpeed_pred'] = 22
dt.loc[dt['cond']=='V3c','targetSpeed_pred'] = 33

coef = list(resultsFit.loc[resultsFit['sub']=='sub-13','coefficients'])[0]
intercept = list(resultsFit.loc[resultsFit['sub']=='sub-13','intercept'])[0]
x = np.arange(-15,60,1)
reg = intercept + coef*x
plt.vlines(x=dt.loc[dt.cond=='V1a','targetSpeed_pred'], ymin=-30, ymax=dt.loc[dt.cond=='V1a','eyeSpeed'], linestyles='dashed', color=colors100['V1a'])
plt.hlines(y=dt.loc[dt.cond=='V1a','eyeSpeed'], xmin=-30, xmax=dt.loc[dt.cond=='V1a','targetSpeed_pred'], linestyles='dashed', color=colors100['V1a'])
plt.vlines(x=dt.loc[dt.cond=='V3d','targetSpeed_pred'], ymin=-30, ymax=dt.loc[dt.cond=='V3d','eyeSpeed'], linestyles='dashed', color=colors100['V3d'])
plt.hlines(y=dt.loc[dt.cond=='V3d','eyeSpeed'], xmin=-30, xmax=dt.loc[dt.cond=='V3d','targetSpeed_pred'], linestyles='dashed', color=colors100['V3d'])
plt.plot(x,reg,c='royalblue')
sns.scatterplot(data=dt, x='targetSpeed_pred',y='eyeSpeed',hue='cond', palette=colors100, s=50, ax=ax2)
plt.xlim([-5,50])
plt.ylim([-1,8])
plt.ylabel('Mean aSPv (°/s)')
plt.xlabel('Estimated Target Speed (°/s)')
ax2.get_legend().set_visible(False)


def normal(mean, std, histmax=False, color="black", label=''):
    x = np.linspace(mean-4*std, mean+4*std, 200)
    p = stats.norm.pdf(x, mean, std)
    if histmax:
        p = p*histmax/max(p)
    z = plt.plot(x, p, c=color, linewidth=1, label=label)

dt = mean_tw_integration.copy()
dt.reset_index(inplace=True)
idxAccel = [idx for idx,row in dt.iterrows() if 'a' in row['condition']]
idxDecel = [idx for idx,row in dt.iterrows() if 'd' in row['condition']]
# ax3 = fig.add_subplot(ax[3:5, 6:10])
ax3 = fig.add_subplot(ax[0, 1])
ax3.set_title('TWI distribution')
sns.histplot(data=dt.loc[idxAccel,:], x='tau', hue='condition',palette=colors100, stat="density")

mean=dt.groupby(['condition']).mean()['tau']
std=dt.groupby(['condition']).std()['tau']
for cond in mean.index:
    if 'a' in cond:
        meanV, stdV = mean.loc[cond], std.loc[cond]
        normal(meanV, stdV, color=colors100[cond], histmax=ax3.get_ylim()[1], label=cond)
plt.xlabel('TWI - last time point')
plt.ylabel('Frequency')
plt.xlim([-2,2])
plt.legend()

ax3 = fig.add_subplot(ax[0, 2])
ax3.set_title('TWI distribution')
sns.histplot(data=dt.loc[idxDecel,:], x='tau', hue='condition',palette=colors100, stat="density")

mean=dt.groupby(['condition']).mean()['tau']
std=dt.groupby(['condition']).std()['tau']
for cond in mean.index:
    if 'd' in cond:
        meanV, stdV = mean.loc[cond], std.loc[cond]
        normal(meanV, stdV, color=colors100[cond], histmax=ax3.get_ylim()[1], label=cond)
plt.xlabel('TWI - last time point')
plt.ylabel('Frequency')
plt.xlim([-2,2])
plt.legend()
# plt.legend(['vacc', 'vdec'])

plt.tight_layout()

plt.savefig('{}/exp2_twi_dist.pdf'.format(output_folder))         
plt.savefig('{}/exp2_twi_dist.png'.format(output_folder))         

for cond in cond100CtrlV2.values():
    if 'c' not in cond:
        print(f'{cond}, tau - ttest against 0:')
        print(pg.ttest(dt.loc[dt.condition.isin([cond]),'tau'], 0, alternative='greater').round(4), '\n')

print("mean: ", mean)
print("std: ", std)



