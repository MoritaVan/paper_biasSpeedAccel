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

main_dir = "../data/biasAcceleration" 
os.chdir(main_dir) 

import warnings
warnings.filterwarnings("ignore")
import traceback

output_folder = '../outputs/exp2'

cm = 1/2.54  # centimeters in inches
single_col = 9*cm
# oneDot5_col = 12.7*cm
two_col = 19*cm

subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06','sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13']
conditions =   [
                'Va-100_V0-0', 
                'Vd-100_V0-0',
                'V1-100_V0-0', 
                'V2-100_V0-0',
                'V3-100_V0-0',
                'V1-75_V3-25',
                'V3-75_V1-25',
                'Va-75_Vd-25',
                'Vd-75_Va-25',
                ]

keys2keep = [
    'sub', 'condition', 'trial',
    'aic_x', 'bic_x', 'chisqr_x', 'redchi_x', 'rmse_x',
    'aic_y', 'bic_y', 'chisqr_y', 'redchi_y', 'rmse_y'
]
data_nonlinear = pd.DataFrame([], columns=keys2keep)
data_linear    = pd.DataFrame([], columns=keys2keep)
data_sigmoid   = pd.DataFrame([], columns=keys2keep)
for idxSub, sub in enumerate(subjects):
    idxSub = idxSub + 1
    print('Subject:',sub)

    for cond in conditions:
        try:
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
        except:
            print('Error!')
            traceback.print_exc()

newColumns_nonlinear = {
    'aic_x': 'aic_x_nonlinear',
    'bic_x': 'bic_x_nonlinear',
    'chisqr_x': 'chisqr_x_nonlinear',
    'redchi_x': 'redchi_x_nonlinear',
    'rmse_x': 'rmse_x_nonlinear',
    'aic_y': 'aic_y_nonlinear',
    'bic_y': 'bic_y_nonlinear',
    'chisqr_y': 'chisqr_y_nonlinear',
    'redchi_y': 'redchi_y_nonlinear',
    'rmse_y': 'rmse_y_nonlinear',
}
newColumns_linear = {
    'aic_x': 'aic_x_linear',
    'bic_x': 'bic_x_linear',
    'chisqr_x': 'chisqr_x_linear',
    'redchi_x': 'redchi_x_linear',
    'rmse_x': 'rmse_x_linear',
    'aic_y': 'aic_y_linear',
    'bic_y': 'bic_y_linear',
    'chisqr_y': 'chisqr_y_linear',
    'redchi_y': 'redchi_y_linear',
    'rmse_y': 'rmse_y_linear',
}
newColumns_sigmoid = {
    'aic_x': 'aic_x_sigmoid',
    'bic_x': 'bic_x_sigmoid',
    'chisqr_x': 'chisqr_x_sigmoid',
    'redchi_x': 'redchi_x_sigmoid',
    'rmse_x': 'rmse_x_sigmoid',
    'aic_y': 'aic_y_sigmoid',
    'bic_y': 'bic_y_sigmoid',
    'chisqr_y': 'chisqr_y_sigmoid',
    'redchi_y': 'redchi_y_sigmoid',
    'rmse_y': 'rmse_y_sigmoid',
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
print('% of >= 0, x axis: ',100*np.mean((data['aic_x_linear']-data['aic_x_nonlinear'])>=0))
print('% of >= 0, y axis: ',100*np.mean((data['aic_y_linear']-data['aic_y_nonlinear'])>=0))
print('AIC: sigmoid model - nonlinear model')
print('% of >= 0, x axis: ',100*np.mean((data['aic_x_sigmoid']-data['aic_x_nonlinear'])>=0))
print('% of >= 0, y axis: ',100*np.mean((data['aic_y_sigmoid']-data['aic_y_nonlinear'])>=0))
print('BIC: linear model - nonlinear model')
print('% of >= 0, x axis: ',100*np.mean((data['bic_x_linear']-data['bic_x_nonlinear'])>=0))
print('% of >= 0, y axis: ',100*np.mean((data['bic_y_linear']-data['bic_y_nonlinear'])>=0))
print('BIC: sigmoid model - nonlinear model')
print('% of >= 0, x axis: ',100*np.mean((data['bic_x_sigmoid']-data['bic_x_nonlinear'])>=0))
print('% of >= 0, y axis: ',100*np.mean((data['bic_y_sigmoid']-data['bic_y_nonlinear'])>=0))
print('RMSE: linear model - nonlinear model')
print('% of >= 0, x axis: ',100*np.mean((data['rmse_x_linear']-data['rmse_x_nonlinear'])>=0))
print('% of >= 0, y axis: ',100*np.mean((data['rmse_y_linear']-data['rmse_y_nonlinear'])>=0))
print('RMSE: sigmoid model - nonlinear model')
print('% of >= 0, x axis: ',100*np.mean((data['rmse_x_sigmoid']-data['rmse_x_nonlinear'])>=0))
print('% of >= 0, y axis: ',100*np.mean((data['rmse_y_sigmoid']-data['rmse_y_nonlinear'])>=0))

plt.figure (figsize=(10,15))
plt.subplot(3,2,1)
plt.title('AIC: linear model - nonlinear model')
sns.histplot(data['aic_x_linear']-data['aic_x_nonlinear'])
plt.subplot(3,2,2)
plt.title('AIC: sigmoid model - nonlinear model')
sns.histplot(data['aic_x_sigmoid']-data['aic_x_nonlinear'])
plt.subplot(3,2,3)
plt.title('BIC: linear model - nonlinear model')
sns.histplot(data['bic_x_linear']-data['bic_x_nonlinear'])
plt.subplot(3,2,4)
plt.title('BIC: sigmoid model - nonlinear model')
sns.histplot(data['bic_x_sigmoid']-data['bic_x_nonlinear'])
plt.subplot(3,2,5)
plt.title('RMSE: linear model - nonlinear model')
sns.histplot(data['rmse_x_linear']-data['rmse_x_nonlinear'])
plt.subplot(3,2,6)
plt.title('RMSE: sigmoid model - nonlinear model')
sns.histplot(data['rmse_x_sigmoid']-data['rmse_x_nonlinear'])
plt.savefig('{}/exp1_ANEMOmodelComparisons_xAxis.pdf'.format(output_folder))
plt.savefig('{}/exp1_ANEMOmodelComparisons_xAxis.png'.format(output_folder))

plt.figure (figsize=(10,15))
plt.subplot(3,2,1)
plt.title('AIC: linear model - nonlinear model')
sns.histplot(data['aic_y_linear']-data['aic_y_nonlinear'])
plt.subplot(3,2,2)
plt.title('AIC: sigmoid model - nonlinear model')
sns.histplot(data['aic_y_sigmoid']-data['aic_y_nonlinear'])
plt.subplot(3,2,3)
plt.title('BIC: linear model - nonlinear model')
sns.histplot(data['bic_y_linear']-data['bic_y_nonlinear'])
plt.subplot(3,2,4)
plt.title('BIC: sigmoid model - nonlinear model')
sns.histplot(data['bic_y_sigmoid']-data['bic_y_nonlinear'])
plt.subplot(3,2,5)
plt.title('RMSE: linear model - nonlinear model')
sns.histplot(data['rmse_y_linear']-data['rmse_y_nonlinear'])
plt.subplot(3,2,6)
plt.title('RMSE: sigmoid model - nonlinear model')
sns.histplot(data['rmse_y_sigmoid']-data['rmse_y_nonlinear'])
plt.savefig('{}/exp1_ANEMOmodelComparisons_yAxis.pdf'.format(output_folder))
plt.savefig('{}/exp1_ANEMOmodelComparisons_yAxis.png'.format(output_folder))