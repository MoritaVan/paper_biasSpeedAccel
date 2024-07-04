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

ctrl_dir = "../data/biasAccelerationControl"
ctrlV2_dir = "../data/biasAccelerationControlV2"

output_folder = '../data/outputs/exp2ctrl'

subjectsCtrl   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']

subjectsCtrlV2 = {
    'sub-01': 'sub-02', # name in controlV2 : new name in control
    'sub-02': 'sub-01', 
    # 'sub-03': 'sub-06', 
    # 'sub-04': 'sub-07', 
    # 'sub-05': 'sub-08'
}

cond100Ctrl =   {
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

condAcc = {
                'Va-100_V0-0': 'Vd-0', 
                'Va-75_Vd-25': 'Vd-30',
                'Vd-75_Va-25': 'Vd-70',
                'Vd-100_V0-0': 'Vd-100',
}

allSubsData = pd.DataFrame([])

for sub in subjectsCtrl:
    h5_file = '{dir}/{s}/{s}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=ctrl_dir,s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-2:])
    data_tmp['sub_txt'] = sub

    data_tmp['cond_OldName'] = data_tmp['condition']
    data_tmp['condition'] = [cond100Ctrl[c] if c in cond100Ctrl.keys() else c for c in data_tmp['condition']]

    data_tmp.loc[data_tmp['SPlat_x'] == data_tmp['aSPon_x']+1, 'aSPon_x'] = np.nan
    data_tmp.loc[data_tmp['SPlat_y'] == data_tmp['aSPon_y']+1, 'aSPon_y'] = np.nan

    allSubsData = pd.concat([allSubsData,data_tmp],ignore_index=True)

for sub in subjectsCtrlV2:
    h5_file = '{dir}/{s}/{s}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=ctrlV2_dir,s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    
    new_subName = subjectsCtrlV2[sub]
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(new_subName[-2:])
    data_tmp['sub_txt'] = new_subName

    data_tmp['cond_OldName'] = data_tmp['condition']
    data_tmp['condition'] = [cond100CtrlV2[c] for c in data_tmp['condition']]

    data_tmp.loc[data_tmp['SPlat_x'] == data_tmp['aSPon_x']+1, 'aSPon_x'] = np.nan
    data_tmp.loc[data_tmp['SPlat_y'] == data_tmp['aSPon_y']+1, 'aSPon_y'] = np.nan

    allSubsData = pd.concat([allSubsData,data_tmp],ignore_index=True)

# Linear Mixed effecs model
# allData variable created above
# exporting to use on jamovi/R

allSubs_cond100 = allSubsData[allSubsData['condition'].isin(cond100CtrlV2.values())]
allSubs_condAcc = allSubsData[allSubsData['condition'].isin(condAcc.keys())]

allSubsData.to_csv('{}/exp2ctrl_params.csv'.format(output_folder))
allSubs_cond100.to_csv('{}/exp2ctrl_params_cond100Pred.csv'.format(output_folder))
allSubs_condAcc.to_csv('{}/exp2ctrl_params_condAccel.csv'.format(output_folder))