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

main_dir = "../data/biasAcceleration"
os.chdir(main_dir)

output_folder = '../outputs/'

subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06',
                'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13'] 

condCte = {
                'V1-100_V0-0': .0, 
                'V1-75_V3-25': .30,
                'V3-75_V1-25': .70,
                'V3-100_V0-0': 1.00,
}

condAcc = {
                'Va-100_V0-0': 0., 
                'Va-75_Vd-25': .30,
                'Vd-75_Va-25': .70,
                'Vd-100_V0-0': 1.00,
}

allSubsData = pd.DataFrame([])

for sub in subjects:
    h5_file = '{s}/{s}_biasAccel_smoothPursuitData_nonlinear.h5'.format(s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-2:])
    data_tmp['sub_txt'] = sub

    # print(data_tmp.head())
        
    allSubsData = pd.concat([allSubsData,data_tmp],ignore_index=True)

# Linear Mixed effecs model
# allData variable created above
# exporting to use on jamovi/R

allSubs_condCte = allSubsData[allSubsData['condition'].isin(condCte)]
allSubs_condAcc = allSubsData[allSubsData['condition'].isin(condAcc)]

allSubs_condCte.to_csv('{}/exp1B/exp1B_params.csv'.format(output_folder))
allSubs_condAcc.to_csv('{}/exp2A/exp2A_params.csv'.format(output_folder))