#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita


"""

import os
import numpy as np
import pandas as pd

main_dir = "./data/biasSpeed"
os.chdir(main_dir)

import warnings
warnings.filterwarnings("ignore")
import traceback

output_folder = '../outputs/exp1'

subjects   = ['s1', 's2', 's3']
subNum     = [int(x[-1]) for x in subjects]

conditions =   [
                'p0', 
                'p10',
                'p25', 
                'p50',
                'p75',
                'p90',
                'p100',
                ]

# read data
print("Reading Data")
for sub in subjects:
    h5_file = '{s}/{s}_biasSpeed_smoothPursuitData.h5'.format(s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    data_tmp.reset_index(inplace=True)
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-1])
    data_tmp['sub_txt'] = sub
    data_tmp['cond_num'] = [float(x[1:])/100 for x in data_tmp['cond']]
    
    data_tmp['ramp_pursuit_x'][data_tmp['ramp_pursuit_x'] > 250]  = np.nan
    data_tmp['ramp_pursuit_x'][data_tmp['ramp_pursuit_x'] < -250] = np.nan
    data_tmp['steady_state_x'][data_tmp['steady_state_x'] > 20] = np.nan
    data_tmp['steady_state_x'][data_tmp['steady_state_x'] < 0]  = np.nan
    
    data_tmp['vel_anti_x'] = data_tmp['velocity_model_x']
    data_tmp['vel_anti_x'][data_tmp['vel_anti_x']==0] = np.nan

    data_tmp['start_anti_x'][data_tmp['latency_x'] == data_tmp['start_anti_x']+1] = np.nan
    
    data_tmp['rotVel_x'] = data_tmp['velocity_model_x']

    # reading the data from the two regressions
    for cond in conditions:
        h5_file2 = '{sub}/{sub}_{cond}_posFilter_classical2.h5'.format(sub=sub, cond=cond) 
        data_tmp2 =  pd.read_hdf(h5_file2, 'data')
        data_tmp2.reset_index(inplace=True)
        
        if cond =='p50': # data files from p50 had wrong names
            new_val = {'HS': "LS", "LS": "HS"}
            data_tmp2.replace({"trial_vel": new_val}, inplace=True)

        idxcond = data_tmp[(data_tmp['cond']==cond)].index
        idxTrial  = data_tmp2[(data_tmp2['trial'].isin(data_tmp.loc[idxcond,'trial']))].index # only trials that passed the quality control

        data_tmp.loc[idxcond,'latency_x']      = list(data_tmp2.loc[idxTrial, 'latency'])
        data_tmp.loc[idxcond,'ramp_pursuit_x'] = list(data_tmp2.loc[idxTrial, 'a_pur']*1000)

    if sub == subjects[0]:
        dataSub = data_tmp.groupby(['cond']).mean()
        allSubsData = data_tmp
    else:
        tmp = data_tmp.groupby(['cond']).mean()
        dataSub = dataSub.append(tmp)
        allSubsData = allSubsData.append(data_tmp)

for sub in subjects:
    print(sub)
    nan_prop_lat = sum(allSubsData.loc[allSubsData['sub_txt']==sub,'latency_x'].isnull())/len(allSubsData.loc[allSubsData['sub_txt']==sub,:])
    nan_prop_pur = sum(allSubsData.loc[allSubsData['sub_txt']==sub,'ramp_pursuit_x'].isnull())/len(allSubsData.loc[allSubsData['sub_txt']==sub,:])
    print('\% excl lat',np.round(nan_prop_lat*100, 2), '\% excl pur', np.round(nan_prop_pur*100, 2))

print(allSubsData.shape)

# Linear Mixed effecs model - fit
# allData created above
allSubsData.to_csv('{}/exp1_params.csv'.formta(output_folder))
