#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita

creates and updates files after running preprocessing.py and qualityctrol.py
"""

#%% bibs
# run always
import os
import numpy as np
import pandas as pd

main_dir = "../data/biasAccelerationControl" 
os.chdir(main_dir)

import warnings
warnings.filterwarnings("ignore")
import traceback


#%% Parameters
# run always
screen_width_px  = 1920 # px
screen_height_px = 1080 # px
screen_width_cm  = 70   # cm
viewingDistance  = 57.  # cm

tan              = np.arctan((screen_width_cm/2)/viewingDistance)
screen_width_deg = 2. * tan * 180/np.pi
px_per_deg       = screen_width_px / screen_width_deg

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


# %% Transform raw data 
# (also save to the same file, under a different folder... 
# if you already run once and have the sXX_4C_rawData.h5 with a rawFormatted subfolder, 
# then you can skip this cell)

keys2save = ['subject','condition', 'trial', 'direction', 'time_x', 'time_y', 
            'posDeg_x', 'posDeg_y', 'velocity_x', 'velocity_y']

float_keys = ['posDeg_x', 'posDeg_y', 'velocity_x', 'velocity_y']
int_keys   = ['trial', 'time_x', 'time_y']

data = dict()
for sub in subjects:
    print('Subject:',sub)
    
    temp = pd.DataFrame()
    for cond in conditions:
        print(cond)
        # read data
        h5_rawfile = '{sub}/{sub}_{cond}_rawData_nonlinear.h5'.format(sub=sub, cond=cond)
        temp  = pd.read_hdf(h5_rawfile,'raw/')

        # get bad data
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(sub=sub, cond=cond) 
        cq        = pd.read_hdf(h5_qcfile, 'data/')

        for index, row in cq.iterrows():
            if (row['keep_trial'] == 0) or (row['good_fit'] == 0): # check if good trial
                temp.drop(temp[temp['trial']==row['trial']].index, inplace=True)
        
        temp.reset_index(inplace=True)
                
        # transform data in a new dataframe
        for index, row in temp.iterrows():
            temp.loc[index]['posDeg_x'][temp.loc[index]['posPxl_x'] < screen_width_px*.05]  = np.nan
            temp.loc[index]['posDeg_x'][temp.loc[index]['posPxl_x'] > screen_width_px*.95]  = np.nan
            temp.loc[index]['posDeg_y'][temp.loc[index]['posPxl_y'] < screen_height_px*.05] = np.nan
            temp.loc[index]['posDeg_y'][temp.loc[index]['posPxl_y'] > screen_height_px*.95] = np.nan

            subj     = np.array(np.arange(len(row['time_x']))).astype(object)
            condi    = np.array(np.arange(len(row['time_x']))).astype(object)
            trial    = np.array(np.arange(len(row['time_x'])))
            tgdir    = np.array(np.arange(len(row['time_x']))).astype(object)
            subj[:]  = sub
            condi[:] = row['condition']
            trial[:] = row['trial']
            tgdir[:] = row['direction']

            newData = np.vstack((subj, condi,trial,tgdir,row['time_x'],row['time_y'],
                        row['posDeg_x'],row['posDeg_y'],row['velocity_x'],row['velocity_y'])).T

            if index == 0:
                data = pd.DataFrame(newData, columns=keys2save)
            else:
                data = data.append(pd.DataFrame(newData, columns=keys2save), ignore_index = True)

        # cast data to correct format
        data[float_keys] = data[float_keys].astype(float)
        data[int_keys]   = data[int_keys].astype(int)

        data.to_hdf(h5_rawfile, 'rawFormatted')
            
# %% read data
# the file for this analysis was generated within the preprocessing script
# run if you don't have the sXX_4C_smoothPursuitData.h5 file

keys = [
        'sub','condition', 'trial', 'original_target_dir', 'trial_velocity',
        'aSPv_x','aSPv_y', 
        'aSPon_x','aSPon_y', 
        'SPlat_x','SPlat_y',
        'SPacc_x','SPacc_y',
        'SPss_x','SPss_y']
params = pd.DataFrame([], columns=keys)

for sub in subjects:
    print('Subject:',sub)

    tempDF = pd.DataFrame()
    for cond in conditions:
        h5_file = '{sub}/{sub}_{cond}_posFilter_nonlinear.h5'.format(sub=sub, cond=cond)
        temp_tmp  = pd.read_hdf(h5_file,'data/')
        
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(sub=sub, cond=cond)
        cq        = pd.read_hdf(h5_qcfile, 'data/')

        for index, row in cq.iterrows():
            if (row['keep_trial'] == 0) or (row['good_fit'] == 0): # check if good trial
                temp_tmp.drop(temp_tmp[temp_tmp['trial']==row['trial']].index, inplace=True)
        temp_tmp.reset_index(drop=True,inplace=True)
               
        tempDF = tempDF.append(temp_tmp, ignore_index=True)

    print('\t',tempDF.shape)

    # transform into a dataframe and save into sXX_4C_smoothPursuitData.h5
    tempDF['sub'] = [sub for _ in range(len(tempDF))]
    newTempDF = tempDF[tempDF.columns.intersection(keys)]

    params = pd.concat([params, newTempDF], ignore_index=True)
    float_keys = ['aSPv_x', 'aSPon_x', 'SPacc_x','SPlat_x','SPss_x',
                  'aSPv_y', 'aSPon_y', 'SPacc_y','SPlat_y','SPss_y']
    params[float_keys] = params[float_keys].astype(float)

    h5_file = ''.join([str(sub),'/', str(sub), '_biasAccel_smoothPursuitData_nonlinear.h5'])
    params.to_hdf(h5_file, 'data')

    del tempDF
