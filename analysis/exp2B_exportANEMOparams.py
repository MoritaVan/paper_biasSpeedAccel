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

ctrl_dir = "../data/biasAccelerationControl_ConstDuration"

output_folder = '../data/outputs/exp2B'

subjectsCtrl   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']

cond100Ctrl =   {
                'V1-100_V0-0': 'V1c', 
                'V2-100_V0-0': 'V2c',
                'V3-100_V0-0': 'V3c',
                'Va-100_V0-0': 'V1a', 
                'Vd-100_V0-0': 'V3d',
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
    data_tmp['condition'] = [cond100Ctrl[c] if c in cond100Ctrl.keys() else condAcc[c] for c in data_tmp['condition']]

    allSubsData = pd.concat([allSubsData,data_tmp],ignore_index=True)


# Linear Mixed effecs model
# allData variable created above
# exporting to use on jamovi/R

allSubs_condAcc = allSubsData[allSubsData['condition'].isin(condAcc.keys())]
allSubs_condAcc.to_csv('{}/exp2B_params.csv'.format(output_folder))