#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita

This script :
- exports data for LMM
"""


import os
import numpy as np
import pandas as pd

main_dir = "../data/biasAccelerationControl"
os.chdir(main_dir)

output_folder = './outputs/exp2ctrl'

subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']

conditions =   [
                'Va-100_V0-0', 
                'Vd-100_V0-0',
                'V1-100_V0-0', 
                'V2-100_V0-0',
                'V3-100_V0-0',
                'Va-75_Vd-25',
                'Vd-75_Va-25'
                ]

cond100 =   {
                'V1-100_V0-0': 'V1', 
                'V2-100_V0-0': 'V2',
                'V3-100_V0-0': 'V3',
                'Va-100_V0-0': 'Va', 
                'Vd-100_V0-0': 'Vd',
}

condAcc = {
                'Va-100_V0-0': 'Vd-0', 
                'Va-75_Vd-25': 'Vd-30',
                'Vd-75_Va-25': 'Vd-70',
                'Vd-100_V0-0': 'Vd-100',
}


def switch_X(argument):
    switcher = {
        'DR': 1,
        'UR': 1,
        'DL': -1,
        'UL': -1
    }
    return switcher.get(argument)

def switch_Y(argument):
    switcher = {
        'DR': -1,
        'UR': 1,
        'DL': -1,
        'UL': 1
    }
    return switcher.get(argument)

for sub in subjects:
    h5_file = '{s}/{s}_biasAccel_smoothPursuitData.h5'.format(s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-2:])
    
    data_tmp['vel_anti_x'] = data_tmp['velocity_model_x']
    data_tmp['vel_anti_y'] = data_tmp['velocity_model_y']
    data_tmp['vel_anti_x'][data_tmp['vel_anti_x']==0] = np.nan
    data_tmp['vel_anti_y'][data_tmp['vel_anti_y']==0] = np.nan

    data_tmp['start_anti_x'][data_tmp['latency_x'] == data_tmp['start_anti_x']+1] = np.nan
    data_tmp['start_anti_y'][data_tmp['latency_y'] == data_tmp['start_anti_y']+1] = np.nan
    
    data_tmp['vel_x'] = data_tmp['velocity_model_x']*switch_X(data_tmp['target_dir'][0])
    data_tmp['vel_y'] = data_tmp['velocity_model_y']*switch_Y(data_tmp['target_dir'][0])

    data_tmp['accel_pursuit_x'] = data_tmp['ramp_pursuit_x']
    data_tmp['accel_pursuit_y'] = data_tmp['ramp_pursuit_y']

    if sub == subjects[0]:
        dataSub = data_tmp.groupby(['cond']).mean()
        allSubsData = data_tmp
    else:
        tmp = data_tmp.groupby(['cond']).mean()
        dataSub = dataSub.append(tmp)
        allSubsData = allSubsData.append(data_tmp)
        

# Linear Mixed effecs model
# allData variable created above
# exporting to use on jamovi/R

allSubs_cond100 = allSubsData[allSubsData['cond'].isin(cond100)]
allSubs_condAcc = allSubsData[allSubsData['cond'].isin(condAcc)]

allSubsData.to_csv('{}/exp2ctrl_params.csv'.format(output_folder))
allSubs_cond100.to_csv('{}/exp2ctrl_params_cond100.csv'.format(output_folder))
allSubs_condAcc.to_csv('{}/exp2ctrl_params_condAcc.csv'.format(output_folder))