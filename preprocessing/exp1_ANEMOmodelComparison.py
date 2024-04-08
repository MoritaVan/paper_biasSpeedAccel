#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vanessa Morita

Compare ANEMO models' performance

"""
#%%
import os
import sys
import h5py
import numpy as np
import pandas as pd
import copy

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append('./')

from functions.utils import *
from functions.updateRC import fontsizeDict,rcConfigDict
plt.rcParams.update(fontsizeDict(small=10,medium=12))
plt.rcParams.update(rcConfigDict(filepath = "./rcparams_config.json"))

main_dir = "../data/biasSpeed" 
os.chdir(main_dir) 

import warnings
warnings.filterwarnings("ignore")
import traceback

output_folder = '../outputs/exp1'


cm = 1/2.54  # centimeters in inches
single_col = 9*cm
# oneDot5_col = 12.7*cm
two_col = 19*cm

subjects   = ['s1', 's2', 's3']
conditions =   [
                'p0',
                'p10',
                'p25',
                'p50',
                'p75',
                'p90',
                'p100',
                ]

keys2keep = [
    'sub', 'condition', 'trial',
    'aic_x', 'bic_x', 'chisqr_x', 'redchi_x', 'rmse_x'
]
data_nonlinear = pd.DataFrame([], columns=keys2keep)
data_linear    = pd.DataFrame([], columns=keys2keep)
data_sigmoid   = pd.DataFrame([], columns=keys2keep)
for idxSub, sub in enumerate(subjects):
    idxSub = idxSub + 1
    print('Subject:',sub)

    for cond in conditions:
        h5_file = '{sub}/{sub}_{cond}_posFilter_nonlinear.h5'.format(sub=sub, cond=cond) 
        tmpDF  = pd.read_hdf(h5_file,'data/')

        tmpDF['sub'] = [sub for _ in range(len(tmpDF))]
        tmpDF = tmpDF[keys2keep]
        data_nonlinear = pd.concat([data_nonlinear,tmpDF], ignore_index=True)

        h5_file = '{sub}/{sub}_{cond}_posFilter_linear.h5'.format(sub=sub, cond=cond) 
        tmpDF  = pd.read_hdf(h5_file,'data/')

        tmpDF['sub'] = [sub for _ in range(len(tmpDF))]
        tmpDF = tmpDF[keys2keep]
        data_linear = pd.concat([data_linear,tmpDF], ignore_index=True)

        h5_file = '{sub}/{sub}_{cond}_posFilter_lin-sigmo.h5'.format(sub=sub, cond=cond) 
        tmpDF  = pd.read_hdf(h5_file,'data/')

        tmpDF['sub'] = [sub for _ in range(len(tmpDF))]
        tmpDF = tmpDF[keys2keep]
        data_sigmoid = pd.concat([data_sigmoid,tmpDF], ignore_index=True)

newColumns_nonlinear = {
    'aic_x': 'aic_nonlinear',
    'bic_x': 'bic_nonlinear',
    'chisqr_x': 'chisqr_nonlinear',
    'redchi_x': 'redchi_nonlinear',
    'rmse_x': 'rmse_nonlinear',
}
newColumns_linear = {
    'aic_x': 'aic_linear',
    'bic_x': 'bic_linear',
    'chisqr_x': 'chisqr_linear',
    'redchi_x': 'redchi_linear',
    'rmse_x': 'rmse_linear',
}
newColumns_sigmoid = {
    'aic_x': 'aic_sigmoid',
    'bic_x': 'bic_sigmoid',
    'chisqr_x': 'chisqr_sigmoid',
    'redchi_x': 'redchi_sigmoid',
    'rmse_x': 'rmse_sigmoid',
}
data_nonlinear.rename(newColumns_nonlinear, axis=1, inplace=True)
data_linear.rename(newColumns_linear, axis=1, inplace=True)
data_sigmoid.rename(newColumns_sigmoid, axis=1, inplace=True)

data_nonlinear.set_index(['sub', 'condition', 'trial'], inplace=True)
data_linear.set_index(['sub', 'condition', 'trial'], inplace=True)
data_sigmoid.set_index(['sub', 'condition', 'trial'], inplace=True)

data = data_nonlinear.join(data_linear)
data = data.join(data_sigmoid)

print('AIC: linear model - nonlinear model')
print('% of >= 0: ',100*np.mean((data['aic_linear']-data['aic_nonlinear'])>=0))
print('AIC: sigmoid model - nonlinear model')
print('% of >= 0: ',100*np.mean((data['aic_sigmoid']-data['aic_nonlinear'])>=0))
print('BIC: linear model - nonlinear model')
print('% of >= 0: ',100*np.mean((data['bic_linear']-data['bic_nonlinear'])>=0))
print('BIC: sigmoid model - nonlinear model')
print('% of >= 0: ',100*np.mean((data['bic_sigmoid']-data['bic_nonlinear'])>=0))
print('RMSE: linear model - nonlinear model')
print('% of >= 0: ',100*np.mean((data['rmse_linear']-data['rmse_nonlinear'])>=0))
print('RMSE: sigmoid model - nonlinear model')
print('% of >= 0: ',100*np.mean((data['rmse_sigmoid']-data['rmse_nonlinear'])>=0))

plt.figure (figsize=(10,15))
plt.subplot(3,2,1)
plt.title('AIC: linear model - nonlinear model')
sns.histplot(data['aic_linear']-data['aic_nonlinear'])
plt.subplot(3,2,2)
plt.title('AIC: sigmoid model - nonlinear model')
sns.histplot(data['aic_sigmoid']-data['aic_nonlinear'])
plt.subplot(3,2,3)
plt.title('BIC: linear model - nonlinear model')
sns.histplot(data['bic_linear']-data['bic_nonlinear'])
plt.subplot(3,2,4)
plt.title('BIC: sigmoid model - nonlinear model')
sns.histplot(data['bic_sigmoid']-data['bic_nonlinear'])
plt.subplot(3,2,5)
plt.title('RMSE: linear model - nonlinear model')
sns.histplot(data['rmse_linear']-data['rmse_nonlinear'])
plt.subplot(3,2,6)
plt.title('RMSE: sigmoid model - nonlinear model')
sns.histplot(data['rmse_sigmoid']-data['rmse_nonlinear'])
plt.savefig('{}/exp1_ANEMOmodelComparisons.pdf'.format(output_folder))
plt.savefig('{}/exp1_ANEMOmodelComparisons.png'.format(output_folder))