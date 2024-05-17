#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  23 12:22:01 2023

@author: Vanessa Morita

"""



#%%
import os
import h5py
import numpy as np
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore")

data_dir = "../data/"

os.chdir(data_dir)

#%%


print('Exp1\n')
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

equation = 'line'
# equation = 'sigmoid'


percent_all=[]
for sub in subjects:
    print('Subject:',sub)

    cq_sub = pd.DataFrame([])
    for cond in conditions:
        try:
            h5_qcfile = '{exp}/{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(exp='biasSpeed',sub=sub, cond=cond)
            cq        = pd.read_hdf(h5_qcfile, 'data/')

            percent_excluded = sum((cq['keep_trial']==0) |(cq['good_fit']==0))
            percent_excluded = percent_excluded/len(cq) * 100
            # print(cond,': ',percent_excluded)
            
            cq_sub = pd.concat([cq_sub,cq])
        except:
            continue

    percent_excluded = sum((cq_sub['keep_trial']==0)|(cq_sub['good_fit']==0))
    percent_excluded = percent_excluded/len(cq_sub) * 100
    print(percent_excluded)

    percent_all.append(percent_excluded)
    
print('median :',np.median(percent_all))
print('max :',np.max(percent_all))
percent_all = sum(percent_all)/len(subjects)
print("mean : ",percent_all)


print('Exp2A\n')

subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06',
                'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13'] 

conditions =   [
                'Va-100_V0-0', 
                'Vd-100_V0-0',
                'V1-100_V0-0', 
                'V2-100_V0-0',
                'V3-100_V0-0',
                'V1-75_V3-25',
                'V3-75_V1-25',
                'Va-75_Vd-25',
                'Vd-75_Va-25'
                ]

percent_all=[]
for sub in subjects:
    print('Subject:',sub)

    cq_sub = pd.DataFrame([])
    for cond in conditions:
        try:
            h5_qcfile = '{exp}/{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(exp='biasAcceleration' , sub=sub, cond=cond)
            cq        = pd.read_hdf(h5_qcfile, 'data/')

            percent_excluded = sum((cq['keep_trial']==0) |(cq['good_fit']==0))
            percent_excluded = percent_excluded/len(cq) * 100
            
            cq_sub = pd.concat([cq_sub,cq])
        except:
            continue

    percent_excluded = sum((cq_sub['keep_trial']==0)|(cq_sub['good_fit']==0))
    percent_excluded = percent_excluded/len(cq_sub) * 100
    print(percent_excluded)

    percent_all.append(percent_excluded)
    
print('median :',np.median(percent_all))
print('max :',np.max(percent_all))
percent_all = sum(percent_all)/len(subjects)
print("mean : ",percent_all)


print('Exp2B\n')

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

percent_all=[]
for sub in subjects:
    print('Subject:',sub)

    cq_sub = pd.DataFrame([])
    for cond in conditions:
        try:
            h5_qcfile = '{exp}/{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(exp='biasAccelerationControl' , sub=sub, cond=cond)
            cq        = pd.read_hdf(h5_qcfile, 'data/')

            percent_excluded = sum((cq['keep_trial']==0) |(cq['good_fit']==0))
            percent_excluded = percent_excluded/len(cq) * 100
            
            cq_sub = pd.concat([cq_sub,cq])
        except:
            continue

    percent_excluded = sum((cq_sub['keep_trial']==0)|(cq_sub['good_fit']==0))
    percent_excluded = percent_excluded/len(cq_sub) * 100
    print(percent_excluded)

    percent_all.append(percent_excluded)
    
print('median :',np.median(percent_all))
print('max :',np.max(percent_all))
percent_all = sum(percent_all)/len(subjects)
print("mean : ",percent_all)