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

exp2B_dir = "../data/biasAccelerationControl_ConstDuration"
exp3_dir  = "../data/biasAccelerationControl_100Pred"

output_folder = '../data/outputs/exp3'

subjectsCtrl   = {'sub-01': 'sub-02', # name in exp2B : new name in exp3
                  'sub-02': 'sub-01'
}

subjectsExp3 = [
    'sub-01', 
    'sub-02', 
    'sub-03', 
    'sub-04', 
    # 'sub-05',
    'sub-06',
    'sub-07',
    'sub-08',
]

cond100Ctrl =   {
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

allSubsData = pd.DataFrame([])

for sub in subjectsCtrl:
    h5_file = '{dir}/{s}/{s}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=exp2B_dir,s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')

    new_subName = subjectsCtrl[sub]
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(new_subName[-2:])
    data_tmp['sub_txt'] = new_subName

    data_tmp['cond_OldName'] = data_tmp['condition']
    data_tmp['condition'] = [cond100Ctrl[c] if c in cond100Ctrl.keys() else c for c in data_tmp['condition']]

    allSubsData = pd.concat([allSubsData,data_tmp],ignore_index=True)

for sub in subjectsExp3:
    h5_file = '{dir}/{s}/{s}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=exp3_dir,s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-2:])
    data_tmp['sub_txt'] = sub

    data_tmp['cond_OldName'] = data_tmp['condition']
    data_tmp['condition'] = [cond100Exp3[c] for c in data_tmp['condition']]

    allSubsData = pd.concat([allSubsData,data_tmp],ignore_index=True)

# Linear Mixed effecs model
# allData variable created above
# exporting to use on jamovi/R

allSubs_cond100 = allSubsData[allSubsData['condition'].isin(cond100Exp3.values())]

allSubs_cond100.to_csv('{}/exp3_params.csv'.format(output_folder))