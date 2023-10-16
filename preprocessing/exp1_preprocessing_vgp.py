#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:22:01 2020

@author: Vanessa Morita


Preprocessing steps

Creates the following files:
subXX_BA_rawData.h5        - raw data for position, velocity and saccades
subXX_BA_qualityControl.h5 - list of trials with labels for bad data and bad fit

"""
#%%
import os
import sys
import h5py
import time as timer
import scipy.io
import numpy as np
import pandas as pd
import copy
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

os.chdir("../ANEMO")
from ANEMO import ANEMO, read_edf
from lmfit import  Model, Parameters


os.chdir("../functions") 
from functions.utils import *

main_dir = "../data/biasSpeed"
os.chdir(main_dir) 

import warnings
warnings.filterwarnings("ignore")
import traceback



#%% ANEMO parameters
screen_width_px  = 1920 # px
screen_height_px = 1080 # px
screen_width_cm  = 70   # cm
viewingDistance  = 57.  # cm

tan              = np.arctan((screen_width_cm/2)/viewingDistance)
screen_width_deg = 2. * tan * 180/np.pi
px_per_deg       = screen_width_px / screen_width_deg

param_exp = {# Mandatory :
                # - number of trials per block :
                      'N_trials' : 1,
                # - number of blocks :
                      'N_blocks' : 1,
                # - direction of the target :
                    # list of lists for each block containing the direction of
                    # the target for each trial is to -1 for left 1 for right
                      'dir_target' : [], # will be defined in the main loop
                # - number of px per degree for the experiment :
                      'px_per_deg' : px_per_deg,
                      'screen_width': screen_width_px,
                      'screen_height': screen_height_px,
              }


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

time_sup = 10 # time window to cut at the end of the trial

sacc_params = {}
# default:
#     'mindur': 5,
#     'maxdur': 100,
#     'minsep': 30,
#     'before_sacc': 5,
#     'after_sacc': 15

sacc_params = {
    1: {
        'mindur': 1,
        'maxdur': 150,
        'minsep': 30,
        'before_sacc': 20,
        'after_sacc': 20
        },
    2: {
        'mindur': 1,
        'maxdur': 150,
        'minsep': 30,
        'before_sacc': 15,
        'after_sacc': 20
        },
    3: {
        'mindur': 1,
        'maxdur': 150,
        'minsep': 30,
        'before_sacc': 15,
        'after_sacc': 25
        },
    4: {
        'mindur': 1,
        'maxdur': 125,
        'minsep': 30,
        'before_sacc': 15,
        'after_sacc': 25
        },
    }

t = np.linspace(0,1,550)
ls = 5*np.ones(len(t))
hs = 15*np.ones(len(t))

#%%

showPlots   = 0 
manualCheck = 0

equation = 'line'

for idxSub, sub in enumerate(subjects):
    idxSub = idxSub + 1
    print('Subject:',sub)


    for cond in conditions:
        paramsSub = pd.DataFrame()
        qualityCtrl = pd.DataFrame()
            
        h5_file = '{sub}/{sub}_{cond}_posFilter_classical2.h5'.format(sub=sub, cond=cond) 
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_classical2.h5'.format(sub=sub, cond=cond) 

        outputFolder_plots = '{sub}/plots/'.format(sub=sub)
        fitPDFFile = '{plotFolder}{sub}_{cond}_fit_classical2.pdf'.format(plotFolder=outputFolder_plots, sub=sub, cond=cond)
        pdf = PdfPages(fitPDFFile) # opens the pdf file to save the figures

        data_dir = '{base_dir}/csv_data/{sub}/{cond}/'.format(base_dir=main_dir, sub=sub, cond=cond)
        data_files = os.listdir(data_dir)

        data_files = np.array([x for x in data_files if 'tr' in x])

        trials = [x[2:] for x in data_files]
        trials = [x.split('_') for x in trials]

        trial_num = np.array([int(x[0]) for x in trials])
        trial_vel = np.array([x[1] for x in trials])

        np.hstack([trial_num, trial_vel, data_files])

        data_filenames = pd.DataFrame([], columns=['trial', 'velocity', 'filename'])
        data_filenames['trial'] = trial_num
        data_filenames['velocity'] = trial_vel
        data_filenames['filename'] = data_files
        data_filenames.sort_values('trial', inplace=True)
        data_filenames.set_index('trial', inplace=True)

        param_exp['N_trials'] = len(data_filenames)

        # change directions from 0/1 diagonals to -1/1
        param_exp['dir_target'] = np.ones(len(data_filenames))
            
        # creates an ANEMO instance
        A   = ANEMO(param_exp)
        Fit = A.Fit(param_exp)

        firstTrial = True
        
        idx_dir_tg = -1
        for index,trial in data_filenames.iterrows():

            idx_dir_tg = idx_dir_tg + 1
            
            print('Trial {0}, cond {1}, sub {2}'.format(index, cond,sub))

            fname = '{folder}/{fname}'.format(folder=data_dir, fname=trial['filename'])
            data = pd.read_csv(fname, sep=',', header=None)
            data.columns=['time', 'pos_x', 'pos_y']           

            velocity_deg_x = A.velocity(data = data['pos_x'],
                                filter_before = True, filter_after = False, cutoff = 30, sample_rate = 1000)

            velocity_deg_y = A.velocity(data = data['pos_y'],
                                filter_before = True, filter_after = False, cutoff = 30, sample_rate = 1000)

            misac = A.detec_misac(velocity_x = velocity_deg_x,
                                velocity_y = velocity_deg_y,
                                t_0        = 0,
                                VFAC       = 5,
                                mindur     = sacc_params[idxSub]['mindur'],
                                maxdur     = sacc_params[idxSub]['maxdur'],
                                minsep     = sacc_params[idxSub]['minsep'])

            new_saccades = []
            [sacc.extend([0,0,0,0,0]) for sacc in misac] # transform misac into the eyelink format
            new_saccades.extend(misac)
            # new_saccades = [x[:2] for x in new_saccades]

            sac = A.detec_sac(velocity_x = velocity_deg_x,
                                velocity_y = velocity_deg_y,
                                t_0        = 0,
                                VFAC       = 5,
                                mindur     = sacc_params[idxSub]['mindur'],
                                maxdur     = sacc_params[idxSub]['maxdur'],
                                minsep     = sacc_params[idxSub]['minsep'])

            [sacc.extend([0,0,0,0,0]) for sacc in sac] # transform misac into the eyelink format
            new_saccades.extend(sac)

            velocity_x_NAN = A.data_NAN(data = velocity_deg_x,
                            saccades          = new_saccades,
                            trackertime       = data.index,
                            before_sacc       = sacc_params[idxSub]['before_sacc'],
                            after_sacc        = sacc_params[idxSub]['after_sacc'])

            velocity_y_NAN = A.data_NAN(data = velocity_deg_y,
                            saccades          = new_saccades,
                            trackertime       = data.index,
                            before_sacc       = sacc_params[idxSub]['before_sacc'],
                            after_sacc        = sacc_params[idxSub]['after_sacc'])

  
            TargetOn = 350

            # test: if bad trial
            if (np.mean(np.isnan(velocity_x_NAN[TargetOn-100:TargetOn+100])) > .7 or
                np.mean(np.isnan(velocity_x_NAN[:-time_sup])) > .6 or
                longestNanRun(velocity_x_NAN[TargetOn-150:TargetOn+600]) > 200):

                print('Skipping bad trial...')

                reason = ''
                if ((np.mean(np.isnan(velocity_x_NAN[TargetOn-100:TargetOn+100])) > .7)):
                    print('too many NaNs around the start of the pursuit')
                    reason = reason + ' >.70 of NaNs around the start of the pursuit'
                if np.mean(np.isnan(velocity_x_NAN[:-time_sup])) > .6:
                    print('too many NaNs overall')
                    reason = reason + ' >{0} of NaNs overall'.format(.6)
                if longestNanRun(velocity_x_NAN[TargetOn-150:TargetOn+600]) > 200:
                    print('at least one nan sequence with more than 200ms')
                    reason = reason + ' At least one nan sequence with more than 200ms'


                newResult = dict()
                newResult['condition']  = cond
                newResult['trial']      = index
                newResult['target_dir'] = 'right'

                newResult['time']                                        = data['time'][:-time_sup]
                newResult['velocity_x'], newResult['velocity_y']         = velocity_x_NAN[:-time_sup], velocity_y_NAN[:-time_sup]
                newResult['saccades']                                    = np.array(new_saccades)

                qCtrl = dict()
                qCtrl['sub']            = sub
                qCtrl['condition']      = cond
                qCtrl['trial']          = index
                qCtrl['keep_trial']     = 0
                qCtrl['good_fit']       = 0
                qCtrl['discard_reason'] = reason

            else: # if not a bad trial, does the fit

                crit = 5 * np.pi/180 # minimum angle between vectors anti and pursuit in radians
                window_size = 50

                time = np.array(data['time'])
                vel_interp = pd.Series(velocity_x_NAN).interpolate()
                vel_interp = np.array(vel_interp)


                a_anti     = 0
                intercept1 = 0
                r2_anti    = 0
                t_start    = 0
                for t in np.arange(70,100):
                    a, inter, r2, _, _ = stats.linregress(time[(data['time']>t)&(data['time']<t+window_size)], vel_interp[(data['time']>t)&(data['time']<t+window_size)])
                    if abs(r2) > abs(r2_anti):
                        r2_anti    = r2
                        a_anti     = a
                        intercept1 = inter
                        t_start    = t

                t_start_pur = 130
                if t_start+window_size >130:
                    t_start_pur = t_start+1
                
                a_pur      = 0
                intercept2 = 0
                r2_pur     = 0
                for t in np.arange(t_start_pur,150):
                    a, inter, r2, _, _  = stats.linregress(time[(data['time']>t)&(data['time']<t+window_size)], vel_interp[(data['time']>t)&(data['time']<t+window_size)])
                    if abs(r2) > abs(r2_pur):
                        r2_pur     = r2
                        a_pur      = a
                        intercept2 = inter

                tw = np.arange(80, 180, 1)
                fitLine1 = a_anti * tw + intercept1
                fitLine2 = a_pur * tw + intercept2
                
                unit_vector_1 = fitLine1 / np.linalg.norm(fitLine1)
                unit_vector_2 = fitLine2 / np.linalg.norm(fitLine2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.arccos(dot_product)

                latency=list()
                if (angle >= crit)&(abs(np.pi-angle) >= crit):
                    idx = np.argwhere(np.isclose(fitLine1, fitLine2, atol=0.1)).reshape(-1)
                    latency = tw[idx]

                if len(latency)==0 : latency = np.nan
                else: latency = latency[0]
                print(latency, angle)
                
                newResult = dict()
                newResult['subject']        = sub
                newResult['condition']      = cond
                newResult['trial']          = index
                newResult['target_dir']     = 'right'
                newResult['trial_velocity'] = trial['velocity']
                newResult['time']           = time
                newResult['velocity_x']     = velocity_x_NAN

                newResult['latency']        = latency
                newResult['a_anti']         = a_anti
                newResult['intercept_anti'] = intercept1
                newResult['a_pur']          = a_pur
                newResult['intercept_pur']  = intercept2

                qCtrl = dict()
                qCtrl['sub']            = sub
                qCtrl['condition']      = cond
                qCtrl['trial']          = index
                qCtrl['keep_trial']     = np.nan
                qCtrl['good_fit']       = np.nan
                qCtrl['discard_reason'] = np.nan

                tmp   = pd.DataFrame([newResult], columns=newResult.keys())
                tmp_qCtrl = pd.DataFrame([qCtrl], columns=qCtrl.keys())

                paramsSub = pd.concat([paramsSub, tmp])
                qualityCtrl = pd.concat([qualityCtrl, tmp_qCtrl])

                fig = plt.figure(figsize=(10, 4))
                plt.suptitle('Trial %d' % index)
                plt.subplot(1,1,1)
                plt.plot(time,velocity_x_NAN)
                plt.plot(np.arange(80, 180, 1),np.arange(80, 180, 1)*a_anti + intercept1, color='g')
                plt.plot(np.arange(80, 180, 1),np.arange(80, 180, 1)*a_pur + intercept2, color='g')
                plt.vlines(x = 0, ymin=0, ymax=16.5, linewidth = 1, linestyle = '--', color = 'k')
                plt.vlines(x = latency, ymin=0, ymax=16.5, linewidth = 1, linestyle = '-', color = 'g')
                plt.xlim(-320,570)
                plt.ylim(-2,17)
                plt.xlabel('Time (ms)')
                plt.ylabel('Velocity - x axis')
                pdf.savefig(fig)
                plt.close()

        paramsSub.to_hdf(h5_file, 'data/')
        qualityCtrl.to_hdf(h5_qcfile, 'data/')

        pdf.close()
