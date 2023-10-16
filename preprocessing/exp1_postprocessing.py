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

import matplotlib.pyplot as plt

os.chdir("/home/vmorita/projects/biasSpeed/functions") # mesocenter
from functions.utils import *


SMALL_SIZE = 10
MEDIUM_SIZE = 12
# BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['font.family'] = 'Helvetica'
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

cm = 1/2.54  # centimeters in inches
single_col = 9*cm
# oneDot5_col = 12.7*cm
two_col = 19*cm

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

data = dict()
for sub in subjects:
    print('Subject:',sub)
    
    temp = pd.DataFrame()
    for cond in conditions:
        print(cond)
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
        
        if cond=='p50': # velocity was wrongly coded for this condition
            temp['velocity'] = ['HS' if x=='LS' else 'LS' for x in temp['velocity']]
                
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

            if index == 0:
                data = pd.DataFrame(newData, columns=keys2save)
            else:
                data = data.append(pd.DataFrame(newData, columns=keys2save), ignore_index = True)

        # cast data to correct format
        data[float_keys] = data[float_keys].astype(float)
        data[int_keys]   = data[int_keys].astype(int)

        data.to_hdf(h5_rawfile, 'rawFormatted')

            
#%% read data
# the file for this analysis was generated within the preprocessing script
# run if you don't have the sXX_4C_smoothPursuitData.h5 file

keys = ['cond', 'trial', 'target_dir', 'trial_vel',
        'velocity_model_x', 
        'velocity_trad_x', 
        'start_anti_x', 
        'ramp_pursuit_x',
        'steady_state_x',
        'classic_lat_x', 
        'latency_x',]

for sub in subjects:
    print('Subject:',sub)

    tempDF = pd.DataFrame()
    for cond in conditions:
        h5_file = '{sub}/{sub}_{cond}_posFilter.h5'.format(sub=sub, cond=cond)
        temp_tmp  = pd.read_hdf(h5_file,'data/')
        
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_afterManualCtrl.h5'.format(sub=sub, cond=cond)
        cq        = pd.read_hdf(h5_qcfile, 'data/')

        for index, row in cq.iterrows():
            if (row['keep_trial'] == 0) or (row['good_fit'] == 0): # check if good trial
                temp_tmp.drop(temp_tmp[temp_tmp['trial']==row['trial']].index, inplace=True)
        temp_tmp.reset_index(inplace=True)
        
        if cond=='p50': # velocity was wrongly coded for this condition
            temp_tmp['trial_velocity'] = ['HS' if x=='LS' else 'LS' for x in temp_tmp['trial_velocity']]

        tempDF = tempDF.append(temp_tmp, ignore_index=True)

    print('\t',tempDF.shape)

    # transform into a dataframe and save into sXX_4C_smoothPursuitData.h5

    temp = np.empty((len(tempDF),len(keys))).astype(object)

    mean_velocity_x = []
    mean_velocity_y = []
    slope_x = []
    slope_y = []
    for index,trial in tempDF.iterrows():
        x = [np.logical_and(t>0,t<=50) for t in trial['time']] # get the time points between 0 and 50 ms
        mean_velocity_x.append(np.nanmean(trial['velocity_x'][x]))

    temp[:,0]  = tempDF['condition']
    temp[:,1]  = tempDF['trial']
    temp[:,2]  = tempDF['target_dir']
    temp[:,3]  = tempDF['trial_velocity']
    temp[:,4]  = tempDF['a_anti_x']*(tempDF['latency_x']-tempDF['start_anti_x'])/1000
    temp[:,5]  = mean_velocity_x
    temp[:,6]  = tempDF['start_anti_x']
    temp[:,7]  = tempDF['ramp_pursuit_x']
    temp[:,8]  = tempDF['steady_state_x']
    temp[:,9]  = tempDF['classic_lat_x']
    temp[:,10] = tempDF['latency_x']

    params = []
    params = pd.DataFrame(temp, columns=keys)

    float_keys = ['velocity_model_x', 
            'velocity_trad_x', 'start_anti_x',
            'ramp_pursuit_x', 'steady_state_x', 
            'classic_lat_x','latency_x',]
    params[float_keys] = params[float_keys].astype(float)

    h5_file = ''.join([str(sub),'/', str(sub), '_biasSpeed_smoothPursuitData.h5'])
    params.to_hdf(h5_file, 'data')

    del tempDF, temp
