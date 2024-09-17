#%%
import sys
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

from sklearn.linear_model import LinearRegression

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

#colorblind friendly
colorAC = np.array([0,158,115,255])/255
colorCT = np.array([114,93,239,255])/255
colorDC = np.array([221,33,125,255])/255


main_dir = "../data/"
# os.chdir(main_dir)

output_folder = "{}/outputs/exp3".format(main_dir)

data_dir_exp3   = "{}/biasAccelerationControl_100Pred".format(main_dir)
data_dir_exp2B = "{}/biasAccelerationControl_ConstDuration".format(main_dir)

subjectsExp2B = [
                'sub-01', 'sub-02'
]
subjectsExp3 = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08']#, 'sub-05']

newSubExp2B = { # subName in Exp2B-fully predictable blocks : corresponding subName in experiment 3
    'sub-01': 'sub-02', 
    'sub-02': 'sub-01', 
}

cond100 =   {
                'V1-100_V0-0': 'V1c', 
                'V2-100_V0-0': 'V2c',
                'V3-100_V0-0': 'V3c',
                'Va-100_V0-0': 'V1a', 
                'Vd-100_V0-0': 'V3d',
}

cond100Exp3 = {
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

    
print('Reading data')
paramsAll = pd.DataFrame()

for idx,sub in enumerate(subjectsExp2B):
    h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_exp2B, sub=sub)
    params  =  pd.read_hdf(h5_file, 'data')
    params  = params[params['condition'].isin(cond100)]
    params['cond'] = [cond100[x] for x in params['condition']]
    params['sub'] = np.ones(params.shape[0])*int(newSubExp2B[sub][-2:])
    paramsAll = pd.concat([paramsAll, params])

for idx,sub in enumerate(subjectsExp3):
    h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_exp3, sub=sub)
    params  =  pd.read_hdf(h5_file, 'data')
    params  = params[params['condition'].isin(cond100Exp3)]
    params['cond'] = [cond100Exp3[x] for x in params['condition']]
    params['sub'] = np.ones(params.shape[0])*int(sub[-2:])
    paramsAll = pd.concat([paramsAll, params])

paramsAll.reset_index(inplace=True)

#%%
# model: perceived velocity for accelerating conditions based on constant conditions

print('Fitting model')
keys2keep = [
    'sub', 'cond', 'trial', 'aSPv_x', 'aSPv_y',
]

dataFit    = pd.DataFrame()
resultsFit = pd.DataFrame()
subjects = subjectsExp3

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
print('Calculating temporal window of integration')

keysTWI  = ['subject', 'condition', 'trial', 'tau', 'pred_targetSpeed']
keysMTWI = ['subject', 'condition', 'tau', 'mean_pred_targetSpeed']
tw_integration = pd.DataFrame([], columns=keysTWI)
mean_tw_integration = pd.DataFrame([], columns=keysMTWI)
for sub in subjects:
    print('Subject:',sub)

    paramsSub = resultsFit[resultsFit['sub']==sub]

    for cond in cond100Exp3.values(): 
        if 'c' not in cond:
            dataFitCond = dataFit[(dataFit['sub']==int(sub[-2:]))&(dataFit['cond']==cond)]
            
            if '1' in cond: v0 = 11
            if '2' in cond: v0 = 22
            if '3' in cond: v0 = 33

            accel = 22 if 'a' in cond else -22

            for trial,row in dataFitCond.iterrows():
                tau = ((row['targetSpeed_pred']-v0)/accel)/2
                tw_integration = pd.concat([tw_integration,pd.DataFrame([[sub,cond,trial, tau, row['targetSpeed_pred']]], columns=keysTWI)], ignore_index=True)

            # twi based on the mean predicted target speed
            meanTgSpeed_test = np.mean(np.array(dataFitCond['targetSpeed_pred']))
            tau = ((meanTgSpeed_test-v0)/accel)/2
            mean_tw_integration = pd.concat([mean_tw_integration,pd.DataFrame([[sub,cond,tau,meanTgSpeed_test]], columns=keysMTWI)], ignore_index=True)

tw_integration.to_csv('{}/exp3_twi.csv'.format(output_folder))

#%%
data2plot = dataFit.groupby(['sub', 'cond', 'trials_type']).mean().reset_index()
data2plot.loc[data2plot['cond']=='V1c','targetSpeed_pred'] = 11
data2plot.loc[data2plot['cond']=='V2c','targetSpeed_pred'] = 22
data2plot.loc[data2plot['cond']=='V3c','targetSpeed_pred'] = 33


colors100 = {
    'V1d': colorDC,
    'V1c': colorCT,
    'V1a': colorAC,
    'V2d': colorDC,
    'V2c': colorCT,
    'V2a': colorAC,
    'V3d': colorDC,
    'V3c': colorCT,
    'V3a': colorAC,
}

coef = resultsFit.coefficients.mean()[0]
intercept = resultsFit.intercept.mean()
x = np.arange(-15,60,1)
reg = intercept + coef*x

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

fig = plt.figure(figsize=(two_col, 10*cm))
# ax = fig.add_gridspec(5, 10)
# ax1 = fig.add_subplot(ax[0:5, 0:5])
ax = fig.add_gridspec(2, 8)

# ax2 = fig.add_subplot(ax[0:2, 6:10])
ax2 = fig.add_subplot(ax[0, 0:2])
ax2.set_title('Individual example')
dt = data2plot[data2plot['sub']==2].copy()
dt.loc[dt['cond']=='V1c','targetSpeed_pred'] = 11
dt.loc[dt['cond']=='V2c','targetSpeed_pred'] = 22
dt.loc[dt['cond']=='V3c','targetSpeed_pred'] = 33

coef = list(resultsFit.loc[resultsFit['sub']=='sub-02','coefficients'])[0]
intercept = list(resultsFit.loc[resultsFit['sub']=='sub-02','intercept'])[0]
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
idxV1 = [idx for idx,row in dt.iterrows() if '1' in row['condition']]
idxV2 = [idx for idx,row in dt.iterrows() if '2' in row['condition']]
idxV3 = [idx for idx,row in dt.iterrows() if '3' in row['condition']]

mean=dt.groupby(['condition']).mean()['tau']
std=dt.groupby(['condition']).std()['tau']

ax3 = fig.add_subplot(ax[0,2:5])
ax3.set_title('TWI distribution')
sns.histplot(data=dt.loc[idxV1,:], x='tau', hue='condition',palette=colors100, stat="probability")
for cond in mean.index:
    if '1' in cond:
        meanV, stdV = mean.loc[cond], std.loc[cond]
        normal(meanV, stdV, color=colors100[cond], histmax=ax3.get_ylim()[1], label=cond)
plt.xlabel('TWI - last time point')
plt.ylabel('Frequency')
plt.xlim([-1.5,1.5])
plt.ylim([0,.35])
plt.legend()

ax3 = fig.add_subplot(ax[0,5:8])
ax3.set_title('TWI distribution')
sns.histplot(data=dt.loc[idxV2,:], x='tau', hue='condition',palette=colors100, stat="probability")
for cond in mean.index:
    if '2' in cond:
        meanV, stdV = mean.loc[cond], std.loc[cond]
        normal(meanV, stdV, color=colors100[cond], histmax=ax3.get_ylim()[1], label=cond)
plt.xlabel('TWI - last time point')
plt.ylabel('Frequency')
plt.xlim([-1.5,1.5])
plt.ylim([0,.35])
plt.legend()

ax3 = fig.add_subplot(ax[1, 5:8])
ax3.set_title('TWI distribution')
sns.histplot(data=dt.loc[idxV3,:], x='tau', hue='condition',palette=colors100, stat="probability")
for cond in mean.index:
    if '3' in cond:
        meanV, stdV = mean.loc[cond], std.loc[cond]
        normal(meanV, stdV, color=colors100[cond], histmax=ax3.get_ylim()[1], label=cond)
plt.xlabel('TWI - last time point')
plt.ylabel('Frequency')
plt.xlim([-1.5,1.5])
plt.ylim([0,.35])
plt.legend()

plt.tight_layout()

plt.savefig('{}/exp3_twi_dist.pdf'.format(output_folder))         
plt.savefig('{}/exp3_twi_dist.png'.format(output_folder))         

#%%
dtExp3 = dt.copy(deep=True).reset_index()


print('Stats')

print('\t\tANOVA')
print(pg.rm_anova(dv='tau', within='condition', data=dtExp3,
               subject='subject', detailed=True), '\n\n')


meanTWI = dtExp3['tau'].mean()
stdTWI = dtExp3['tau'].std()

print(f'mean TWI: {meanTWI}, std: {stdTWI}')