#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita

creates and updates files after running preprocessing.py
*must run the qualitycontrol.py before 
"""

#%% bibs
# run always
import os
import numpy as np
import pandas as pd

main_dir = "../data/biasSpeed" 
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

tan               = np.arctan((screen_width_cm/2)/viewingDistance)
screen_width_deg  = 2. * tan * 180/np.pi
px_per_deg        = screen_width_px / screen_width_deg
screen_height_deg = screen_height_px / px_per_deg

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

# %% Transform raw data 


keys2save = ['subject','condition', 'trial', 'tg_dir', 'tg_vel', 'time', 
            'posDeg_x', 'posDeg_y', 'velocity_x', 'velocity_y']

float_keys = ['posDeg_x', 'posDeg_y', 'velocity_x', 'velocity_y']
int_keys   = ['trial', 'time']

for sub in subjects:
    print('Subject:',sub)
    
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
        
        if cond=='p50': # velocity was wrongly coded for this condition
            temp['velocity'] = ['HS' if x=='LS' else 'LS' for x in temp['velocity']]
                
        data = pd.DataFrame([], columns=keys2save)
        # transform data in a new dataframe
        for index, row in temp.iterrows():
            temp.loc[index]['posDeg_x'][temp.loc[index]['posDeg_x'] < screen_width_deg*.05]  = np.nan
            temp.loc[index]['posDeg_x'][temp.loc[index]['posDeg_x'] > screen_width_deg*.95]  = np.nan
            temp.loc[index]['posDeg_y'][temp.loc[index]['posDeg_y'] < screen_height_deg*.05] = np.nan
            temp.loc[index]['posDeg_y'][temp.loc[index]['posDeg_y'] > screen_height_deg*.95] = np.nan

            subj     = np.array(np.arange(len(row['time']))).astype(object)
            condi    = np.array(np.arange(len(row['time']))).astype(object)
            trial    = np.array(np.arange(len(row['time'])))
            tgdir    = np.array(np.arange(len(row['time']))).astype(object)
            tgvel    = np.array(np.arange(len(row['time']))).astype(object)
            subj[:]  = sub
            condi[:] = row['condition']
            trial[:] = row['trial']
            tgdir[:] = row['direction']
            tgvel[:] = row['velocity']

            newData = np.vstack((subj, condi,trial,tgdir,tgvel,row['time'],
                        row['posDeg_x'],row['posDeg_y'],row['velocity_x'],row['velocity_y'])).T

            data = pd.concat([data,pd.DataFrame(newData, columns=keys2save)], ignore_index = True)

        # cast data to correct format
        data[float_keys] = data[float_keys].astype(float)
        data[int_keys]   = data[int_keys].astype(int)

        data.to_hdf(h5_rawfile, 'rawFormatted')
    
    del data, newData

            
#%% read data
# the file for this analysis was generated within the preprocessing script
# run if you don't have the sXX_4C_smoothPursuitData.h5 file
keys = ['sub','condition', 'trial', 'target_dir', 'trial_velocity',
        'aSPv', 
        'aSPon', 
        'SPlat',
        'SPacc',
        'SPss']
for sub in subjects:
    print('Subject:',sub)
    params = pd.DataFrame([], columns=keys)
    for cond in conditions:
        h5_file = '{sub}/{sub}_{cond}_posFilter_nonlinear.h5'.format(sub=sub, cond=cond)
        temp    = pd.read_hdf(h5_file,'data/')

        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(sub=sub, cond=cond) 
        cq        = pd.read_hdf(h5_qcfile, 'data/')

        for index, row in cq.iterrows():
            if (row['keep_trial'] == 0) or (row['good_fit'] == 0): # check if good trial
                temp.drop(temp[temp['trial']==row['trial']].index, inplace=True)
        temp.reset_index(inplace=True)
        
        if cond=='p50': # velocity was wrongly coded for this condition
            temp['trial_velocity'] = ['HS' if x=='LS' else 'LS' for x in temp['trial_velocity']]

        temp['cond'] = cond
        temp['sub'] = sub
        params = pd.concat([params, temp], ignore_index=True)

    # transform into a dataframe and save into sXX_biasSpeed_smoothPursuitData.h5
    params = params[params.columns.intersection(keys)]

    float_keys = ['aSPv', 'aSPon',
            'SPacc','SPlat','SPss']
    params[float_keys] = params[float_keys].astype(float)

    h5_file = '{s}/{s}_biasSpeed_smoothPursuitData_nonlinear.h5'.format(s=sub)
    params.to_hdf(h5_file, 'data')

    del temp
