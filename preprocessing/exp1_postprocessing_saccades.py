#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  25 2023

@author: Vanessa Morita

creates saccades df
"""

#%% bibs
# run always
import os
import numpy as np
import pandas as pd

main_dir = "../biasSpeed"
data_dir = "../data/biasSpeed" # 
os.chdir(main_dir) 

import warnings
warnings.filterwarnings("ignore")
import traceback


screen_width_px  = 1920 # px
screen_height_px = 1080 # px
screen_width_cm  = 70   # cm
viewingDistance  = 57.  # cm

tan              = np.arctan((screen_width_cm/2)/viewingDistance)
screen_width_deg = 2. * tan * 180/np.pi
px_per_deg       = screen_width_px / screen_width_deg

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


equation = 'line'

# %% Transform raw data 


print('Creating first saccade dataframe')

keys2drop = ['time_x','time_y',
            'posDeg_x', 'posDeg_y',
            'posPxl_x', 'posPxl_y',
            'velocity_x', 'velocity_y']

saccKeys  = ['sub','condition', 'trial', 'direction','tg_vel','sacc_idx',
            'time_start', 'time_end', 'duration',
            'x_start', 'y_start', 'x_end', 'y_end',
            'amplitude'] 

float_keys = ['x_start', 'y_start', 'x_end', 'y_end', 'amplitude','time_start', 'time_end', 'duration']
int_keys   = ['trial', 'sacc_idx']

df_all = pd.DataFrame([], columns=saccKeys)
data = dict()
for sub in subjects:
    print('Subject:',sub)
    h5_file = '{dir}/{sub}/{sub}_biasSpeed_smoothPursuitData.h5'.format(dir=data_dir, sub=sub)
    smoothData = pd.read_hdf(h5_file, 'data')
    smoothData = smoothData[['cond', 'trial', 'trial_vel']]
    print(smoothData.cond.unique())
    
    print("Reading data")
    temp = pd.DataFrame()
    for cond in conditions:
        print(cond)
        try:
            # read data
            h5_rawfile = '{dir}/{sub}/{sub}_{cond}_rawData.h5'.format(dir=data_dir, sub=sub, cond=cond)
            temp_raw  = pd.read_hdf(h5_rawfile,'raw/')

            # get bad data
            h5_qcfile = '{dir}/{sub}/{sub}_{cond}_qualityControl_afterManualCtrl.h5'.format(dir=data_dir, sub=sub, cond=cond)
            cq        = pd.read_hdf(h5_qcfile, 'data/')

            for index, row in cq.iterrows():
                if (row['keep_trial'] == 0) or (row['good_fit'] == 0): # check if good trial
                    temp_raw.drop(temp_raw[temp_raw['trial']==row['trial']].index, inplace=True)
            # temp_raw.drop(keys2drop, axis=1, inplace=True)


            idx = smoothData['cond']==cond
            
            temp_raw['tg_vel'] = list(smoothData.loc[idx,'trial_vel'])
            if cond=='p50': # velocity was wrongly coded for this condition
                temp_raw['tg_vel'] = ['HS' if x=='LS' else 'LS' for x in temp_raw['tg_vel']]
                
            temp_raw.reset_index(inplace=True)
            temp = pd.concat((temp,temp_raw), ignore_index=True)

        except:
            print("Condition not found")
            traceback.print_exc()

    print('Creating new dataframe')
    data_sub = pd.DataFrame([], columns=saccKeys)
    for index, row in temp.iterrows():
        saccades = [sacc for sacc in row['saccades']]
        saccades.sort()
        saccades = [s for s in saccades if (s[0]>60)&(s[0]<500)] # I only want visually guided saccades and do not want too late saccades
        

        if len(saccades) > 0:

            sac_idx = 0
            while sac_idx < len(saccades)-1:
                if (saccades[sac_idx][0] <= saccades[sac_idx+1][0]) & (saccades[sac_idx][1] >= saccades[sac_idx+1][1]):
                    del saccades[sac_idx+1]
                else:
                    sac_idx += 1

            for idx,sac in enumerate(saccades):
                time_start = sac[0]
                time_end   = sac[1]
                duration   = sac[2] if sac[2]!=0 else time_end-time_start
                x_start    = sac[3] / px_per_deg
                y_start    = sac[4] / px_per_deg
                x_end      = sac[5] / px_per_deg
                y_end      = sac[6] / px_per_deg

                if x_start == 0:
                    idx_start = row['time']==time_start
                    idx_end   = row['time']==time_end

                    x_start = list(row['posDeg_x'][idx_start])[0] if len(row['posDeg_x'][idx_start]) else np.nan
                    x_end   = list(row['posDeg_x'][idx_end])[0] if len(row['posDeg_x'][idx_end]) else np.nan
                    y_start = list(row['posDeg_y'][idx_start])[0] if len(row['posDeg_y'][idx_start]) else np.nan
                    y_end   = list(row['posDeg_y'][idx_end])[0] if len(row['posDeg_y'][idx_end]) else np.nan

                amplitude = np.round(np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2),4)

                newData   = [(sub,row['condition'],row['trial'],row['direction'],row['tg_vel'],idx,
                            time_start,time_end,duration,
                            x_start,y_start,x_end,
                            y_end,amplitude)]
                
                data_sub = data_sub.append(pd.DataFrame(newData, columns=saccKeys), ignore_index = True)
                
        else:
            newData   = [(sub,row['condition'],row['trial'],row['direction'],row['tg_vel'],0,
                        np.nan,np.nan,np.nan,
                        np.nan,np.nan,np.nan,
                        np.nan,np.nan)]
            data_sub = data_sub.append(pd.DataFrame(newData, columns=saccKeys), ignore_index = True)

    # cast data to correct format
    data_sub[float_keys] = data_sub[float_keys].astype(float)
    data_sub[int_keys]   = data_sub[int_keys].astype(int)
    
    h5_file = '{dir}/{sub}/{sub}_biasSpeed_saccadeData.h5'.format(dir=data_dir, sub=sub)
    data_sub.to_hdf(h5_file, 'data')

    df_all = df_all.append(data_sub, ignore_index = True)

print(df_all.sample(10))
print(df_all.head(10))
df_all.to_csv('{data_dir}/dadosSaccade_allSubs_biasSpeed.csv'.format(data_dir=data_dir))