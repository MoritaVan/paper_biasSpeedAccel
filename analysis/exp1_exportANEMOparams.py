#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita


"""

import os
import numpy as np
import pandas as pd

main_dir = "../data/biasSpeed"
os.chdir(main_dir)

output_folder = './outputs'

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

allSubsData = pd.DataFrame([])
for sub in subjects:
    h5_file = '{s}/{s}_biasSpeed_smoothPursuitData_no-SPss.h5'.format(s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    data_tmp.reset_index(inplace=True)
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-1])
    data_tmp['sub_txt'] = sub
    data_tmp['cond_num'] = [float(x[1:])/100 for x in data_tmp['cond']]
    
    data_tmp.loc[data_tmp['SPacc'] > 250, 'SPacc']  = np.nan
    data_tmp.loc[data_tmp['SPacc'] < -250, 'SPacc'] = np.nan

    data_tmp.loc[data_tmp['SPlat'] == data_tmp['aSPon']+1, 'aSPon'] = np.nan

    # reading the data from the two regressions
    # for cond in conditions:
    #     h5_file2 = '{sub}/{sub}_{cond}_posFilter_classical2.h5'.format(sub=sub, cond=cond) 
    #     data_tmp2 =  pd.read_hdf(h5_file2, 'data')
    #     data_tmp2.reset_index(inplace=True)
        
    #     if cond =='p50': # data files from p50 had wrong names
    #         new_val = {'HS': "LS", "LS": "HS"}
    #         data_tmp2.replace({"trial_vel": new_val}, inplace=True)

    #     idxcond = data_tmp[(data_tmp['cond']==cond)].index
    #     idxTrial  = data_tmp2[(data_tmp2['trial'].isin(data_tmp.loc[idxcond,'trial']))].index # only trials that passed the quality control

    #     data_tmp.loc[idxcond,'latency_x']      = list(data_tmp2.loc[idxTrial, 'latency'])
    #     data_tmp.loc[idxcond,'ramp_pursuit_x'] = list(data_tmp2.loc[idxTrial, 'a_pur']*1000)

    allSubsData = pd.concat([allSubsData,data_tmp],ignore_index=True)

# for sub in subjects:
#     print(sub)
#     nan_prop_lat = sum(allSubsData.loc[allSubsData['sub_txt']==sub,'latency_x'].isnull())/len(allSubsData.loc[allSubsData['sub_txt']==sub,:])
#     nan_prop_pur = sum(allSubsData.loc[allSubsData['sub_txt']==sub,'ramp_pursuit_x'].isnull())/len(allSubsData.loc[allSubsData['sub_txt']==sub,:])
#     print('\% excl lat',np.round(nan_prop_lat*100, 2), '\% excl pur', np.round(nan_prop_pur*100, 2))

print(allSubsData.shape)

# Linear Mixed effecs model - fit
# allData created above
allSubsData.to_csv('{}/exp1_params.csv'.format(output_folder))
