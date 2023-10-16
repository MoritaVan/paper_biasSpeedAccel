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

import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

main_dir = "../data/biasAcceleration" # mesocenter
os.chdir(main_dir) # pc lab

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

subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06',
                'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12','sub-13']

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

equation = 'line'

# %% Transform raw data 
# (also save to the same file, under a different folder... 

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
        try:
            # read data
            h5_rawfile = '{sub}/{sub}_{cond}_rawData.h5'.format(sub=sub, cond=cond)
            temp  = pd.read_hdf(h5_rawfile,'raw/')

            # get bad data
            h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_afterManualCtrl.h5'.format(sub=sub, cond=cond)
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
        except Exception as e:
            print('Error! \n Couldn\'t process {}, condition {}'.format(sub,cond))
            traceback.print_exc()
            
#%% read data
# the file for this analysis was generated within the preprocessing script
# run if you don't have the sXX_4C_smoothPursuitData.h5 file

keys = ['cond', 'trial', 'target_dir', 'trial_vel',
        'velocity_model_x', 'velocity_model_y', 
        'velocity_trad_x', 'velocity_trad_y', 
        'start_anti_x', 'start_anti_y',
        'ramp_pursuit_x', 'ramp_pursuit_y', 
        'steady_state_x', 'steady_state_y',
        'classic_lat_x', 'classic_lat_y',
        'latency_x', 'latency_y']

for sub in subjects:
    print('Subject:',sub)

    tempDF = pd.DataFrame()
    for cond in conditions:
        try:
            h5_file = '{sub}/{sub}_{cond}_posFilter.h5'.format(sub=sub, cond=cond)
            temp_tmp  = pd.read_hdf(h5_file,'data/')

            h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_afterManualCtrl.h5'.format(sub=sub, cond=cond)
            cq        = pd.read_hdf(h5_qcfile, 'data/')

            for index, row in cq.iterrows():
                if (row['keep_trial'] == 0) or (row['good_fit'] == 0): # check if good trial
                    temp_tmp.drop(temp_tmp[temp_tmp['trial']==row['trial']].index, inplace=True)
            temp_tmp.reset_index(inplace=True)


            if equation=='line':
                temp_tmp['new_latency_x'] = np.zeros(len(temp_tmp))
                temp_tmp['new_latency_y'] = np.zeros(len(temp_tmp))

            tempDF = tempDF.append(temp_tmp, ignore_index=True)
        except Exception as e:
            print('Error! \n Couldn\'t process {}, condition {}'.format(sub,cond))
            traceback.print_exc()

    print('\t',tempDF.shape)

    # transform into a dataframe and save into sXX_4C_smoothPursuitData.h5

    temp = np.empty((len(tempDF),len(keys))).astype(object)

    mean_velocity_x = []
    mean_velocity_y = []
    slope_x = []
    slope_y = []
    for index,trial in tempDF.iterrows():
#         print(trial)
        x = [np.logical_and(t>0,t<=50) for t in trial['time_x']] # get the time points between 0 and 50 ms
        mean_velocity_x.append(np.nanmean(trial['velocity_x'][x]))
        mean_velocity_y.append(np.nanmean(trial['velocity_y'][x]))

    temp[:,0]  = tempDF['condition']
    temp[:,1]  = tempDF['trial']
    temp[:,2]  = tempDF['target_dir']
    temp[:,3]  = tempDF['trial_velocity']
    temp[:,4]  = tempDF['a_anti_x']*(tempDF['latency_x']-tempDF['start_anti_x'])/1000
    temp[:,5]  = tempDF['a_anti_y']*(tempDF['latency_y']-tempDF['start_anti_y'])/1000
    temp[:,6]  = mean_velocity_x
    temp[:,7]  = mean_velocity_y
    temp[:,8]  = tempDF['start_anti_x']
    temp[:,9]  = tempDF['start_anti_y']
    temp[:,10]  = tempDF['ramp_pursuit_x']
    temp[:,11] = tempDF['ramp_pursuit_y']
    temp[:,12] = tempDF['steady_state_x']
    temp[:,13] = tempDF['steady_state_y']
    temp[:,14] = tempDF['classic_lat_x']
    temp[:,15] = tempDF['classic_lat_y']
    temp[:,16]  = tempDF['latency_x']
    temp[:,17]  = tempDF['latency_y']

    params = []
    params = pd.DataFrame(temp, columns=keys)

    float_keys = ['velocity_model_x', 'velocity_model_y', 
            'velocity_trad_x', 'velocity_trad_y', 'start_anti_x', 'start_anti_y',
            'ramp_pursuit_x', 'ramp_pursuit_y', 'steady_state_x', 'steady_state_y', 
            'classic_lat_x', 'classic_lat_y','latency_x', 'latency_y']
    params[float_keys] = params[float_keys].astype(float)

    h5_file = ''.join([str(sub),'/', str(sub), '_biasAccel_smoothPursuitData.h5'])
    params.to_hdf(h5_file, 'data')

    del tempDF, temp
