#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:22:01 2020

@author: Vanessa Morita


Preprocessing steps

Creates the following files:
subXX_BA_posFilter.h5      - data generated by ANEMO
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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

os.chdir("../ANEMO") 
from ANEMO import ANEMO, read_edf

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
        if cond != conditions[0]: break

        if equation=='sigmoid':
            allow_baseline, allow_horizontalShift, allow_acceleration = False, True, False
        else:
            fit_steadyState = True
            
        h5_file = '{sub}/{sub}_{cond}_posFilter_noConstraint.h5'.format(sub=sub, cond=cond) 
        h5_rawfile = '{sub}/{sub}_{cond}_rawData_noConstraint.h5'.format(sub=sub, cond=cond)
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_noConstraint.h5'.format(sub=sub, cond=cond) 

        paramsRaw   = pd.DataFrame()
        qualityCtrl = pd.DataFrame()
        paramsSub   = pd.DataFrame()

        outputFolder_plots = '{sub}/plots/'.format(sub=sub)
        fitPDFFile = '{plotFolder}{sub}_{cond}_fit_noConstraint.pdf'.format(plotFolder=outputFolder_plots, sub=sub, cond=cond)

        # make figure folder
        try: os.makedirs(outputFolder_plots)
        except: pass
        # make figure folder
        try: os.makedirs(''.join([outputFolder_plots,'qc/']))
        except: pass

        # for diagnostic purposes:
        nanOverallFile  = '{}/qc/{}_{}_nanOverall.pdf'.format(outputFolder_plots, sub, cond)
        nanOnsetFile    = '{}/qc/{}_{}_nanOnset.pdf'.format(outputFolder_plots, sub, cond)
        nanSequenceFile = '{}/qc/{}_{}_nanSequence.pdf'.format(outputFolder_plots, sub, cond)

        nanOverallpdf   = PdfPages(nanOverallFile)
        nanOnsetpdf     = PdfPages(nanOnsetFile)
        nanSequencepdf  = PdfPages(nanSequenceFile)

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

            pos_deg_x = data['pos_x']
            pos_deg_y = data['pos_y']
            
            # calc saccades relative to t_0
            # because I am passing the time relative to t_0 to ANEMO   
            sacc_save = copy.deepcopy(new_saccades)
            for sacc in sacc_save:
                sacc[0] = sacc[0] - 350
                sacc[1] = sacc[1] - 350
                
                
            sDict = {
                'condition': cond,
                'trial': index,
                'direction': 'right',
                'velocity': trial['velocity'],
                'time': data['time'],
                'posDeg_x': pos_deg_x,
                'posDeg_y': pos_deg_y,
                'velocity_x': velocity_x_NAN,
                'velocity_y': velocity_y_NAN,
                'saccades': sacc_save
            }

            # save trial data to a dataframe
            paramsRaw = pd.concat([paramsRaw, pd.DataFrame([sDict], columns=sDict.keys())], ignore_index=True)

            TargetOn = 350

            # test: if bad trial
            if (np.mean(np.isnan(velocity_x_NAN[TargetOn-100:TargetOn+100])) > .7 or
                np.mean(np.isnan(velocity_x_NAN[:-time_sup])) > .6 or
                longestNanRun(velocity_x_NAN[TargetOn-150:TargetOn+600]) > 200):

                print('Skipping bad trial...')

                plt.clf()
                fig = plt.figure(figsize=(10, 4))
                plt.suptitle('Trial %d' % index)
                plt.subplot(1,2,1)
                plt.plot(velocity_x_NAN[:-time_sup])
                plt.axvline(x = -350, linewidth = 1, linestyle = '--', color = 'k')
                plt.axvline(x = 0, linewidth = 1, linestyle = '--', color = 'k')
                plt.xlim(-320,570)
                plt.ylim(-2,17)
                plt.xlabel('Time (ms)')
                plt.ylabel('Velocity - x axis')
                plt.subplot(1,2,2)
                plt.plot(velocity_y_NAN[:-time_sup])
                plt.axvline(x = -350, linewidth = 1, linestyle = '--', color = 'k')
                plt.axvline(x = 0, linewidth = 1, linestyle = '--', color = 'k')
                plt.xlim(-320,570)
                plt.ylim(-2,17)
                plt.xlabel('Time (ms)')
                plt.ylabel('Velocity - y axis')

                reason = ''
                if ((np.mean(np.isnan(velocity_x_NAN[TargetOn-100:TargetOn+100])) > .7)):
                    print('too many NaNs around the start of the pursuit')
                    reason = reason + ' >.70 of NaNs around the start of the pursuit'
                    nanOnsetpdf.savefig(fig)
                if np.mean(np.isnan(velocity_x_NAN[:-time_sup])) > .6:
                    print('too many NaNs overall')
                    reason = reason + ' >{0} of NaNs overall'.format(.6)
                    nanOverallpdf.savefig(fig)
                if longestNanRun(velocity_x_NAN[TargetOn-150:TargetOn+600]) > 200:
                    print('at least one nan sequence with more than 200ms')
                    reason = reason + ' At least one nan sequence with more than 200ms'
                    nanSequencepdf.savefig(fig)


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

                velocity_x_NAN = velocity_x_NAN[data['time'] <= 300]
                pos_x = data['pos_x'][data['time'] <= 300]
                time = data['time'][data['time'] <= 300]

                classic_lat_x, classic_max_x, classic_ant_x = A.classical_method.Full(velocity_x_NAN, 350)
                # print('classical latency: {:+.2f}, max: {:+.2f}, anti: {:+.2f}'.format(classic_lat_x, classic_max_x, classic_ant_x))
                classic_ant = classic_ant_x if not np.isnan(classic_ant_x) else 0.5

                param_fit, inde_var = Fit.generation_param_fit(equation = 'fct_velocity_line',
                                                    dir_target    = param_exp['dir_target'][idx_dir_tg],
                                                    trackertime   = time.index.values,
                                                    TargetOn      = 350,
                                                    StimulusOf    = 0,
                                                    saccades      = new_saccades,
                                                    value_latency = classic_lat_x,
                                                    value_steady_state = classic_max_x,
                                                    value_anti    = classic_ant*5)
                
                velocity_x_NAN[:15]  = np.nan
                velocity_x_NAN[-15:] = np.nan
                
                result_x = Fit.Fit_trial(velocity_x_NAN,
                                        equation      = 'fct_velocity_line',
                                        data_x        = pos_x,
                                        dir_target    = param_exp['dir_target'][idx_dir_tg],
                                        trackertime   = time.index.values,
                                        TargetOn      = 350,
                                        StimulusOf    = 0,
                                        saccades      = new_saccades,
                                        time_sup      = None,
                                        param_fit     = param_fit,
                                        inde_vars     = inde_var,
                                        value_latency = classic_lat_x,
                                        value_maxi    = classic_max_x,
                                        value_anti    = classic_ant*5,
                                        fit_steadyState = fit_steadyState,
                                        )

                eq_x_tmp = ANEMO.Equation.fct_velocity_line(inde_var['x'],
                                            result_x.params['dir_target'],
                                            result_x.params['baseline'],
                                            result_x.params['start_anti'],
                                            result_x.params['a_anti'],
                                            result_x.params['latency'],
                                            result_x.params['ramp_pursuit'],
                                            result_x.params['do_whitening'],
                                            fit_steadyState = fit_steadyState)

                eq_x_tmp = np.array(eq_x_tmp)
                eq_x = np.zeros(len(time))
                eq_x[:] = np.nan
                eq_x[:len(eq_x_tmp)] = eq_x_tmp

                newResult = dict()
                newResult['condition']      = cond
                newResult['trial']          = index
                newResult['target_dir']     = 'right'
                newResult['trial_velocity'] = trial['velocity']
                newResult['time']           = time
                newResult['velocity_x']     = velocity_x_NAN
                newResult['saccades']       = np.array(new_saccades)

                newResult['dir_target_x']   = result_x.params['dir_target'].value               
                newResult['baseline_x']     = result_x.params['baseline'].value             
                newResult['start_anti_x']   = result_x.params['start_anti'].value-350
                newResult['a_anti_x']       = result_x.params['a_anti'].value           
                newResult['latency_x']      = result_x.params['latency'].value-350
                newResult['ramp_pursuit_x'] = result_x.params['ramp_pursuit'].value
                newResult['do_whitening_x'] = result_x.params['do_whitening'].value

                if equation=='sigmoid':
                    newResult['horizontal_shift_x'] = result_x.params['horizontal_shift'].value
                    newResult['allow_baseline'] = allow_baseline
                    newResult['allow_horizontalShift'] = allow_horizontalShift
                    newResult['allow_acceleration'] = allow_acceleration
                else:
                    newResult['fit_steadyState'] = fit_steadyState


                newResult['aic_x']         = result_x.aic
                newResult['bic_x']         = result_x.bic
                newResult['chisqr_x']      = result_x.chisqr
                newResult['redchi_x']      = result_x.redchi
                newResult['residual_x']    = result_x.residual
                newResult['rmse_x']        = np.sqrt(np.mean([x*x for x in result_x.residual]))

                newResult['classic_lat_x'] = classic_lat_x
                newResult['classic_max_x'] = classic_max_x
                newResult['classic_ant_x'] = classic_ant_x

                newResult['fit_x'] = eq_x

                target_time = t
                target_vel  = ls if trial['velocity']=='LS' else hs 

                f = plotFig(index, target_time, target_vel,
                    newResult['time'], newResult['velocity_x'], eq_x, newResult['start_anti_x'], newResult['latency_x'],
                    show=showPlots)

                pdf.savefig(f)
                plt.close(f)

                qCtrl = dict()
                qCtrl['sub']            = sub
                qCtrl['condition']      = cond
                qCtrl['trial']          = index

                if newResult['rmse_x'] > 10:
                    qCtrl['keep_trial']     = np.nan
                    qCtrl['good_fit']       = 0
                    qCtrl['discard_reason'] = np.nan
                elif manualCheck:
                    qCtrl['keep_trial']     = int(input("Keep trial? \npress (1) to keep or (0) to discard\n "))
                    while qCtrl['keep_trial'] != 0 and qCtrl['keep_trial'] != 1:
                        qCtrl['keep_trial'] = int(input("Keep trial? \npress (1) to keep or (0) to discard\n "))

                    qCtrl['good_fit']       = int(input("Good fit? \npress (1) for yes or (0) for no\n "))
                    while qCtrl['good_fit'] != 0 and qCtrl['keep_trial'] != 1:
                        qCtrl['good_fit']   = int(input("Good fit? \npress (1) for yes or (0) for no\n "))

                    qCtrl['discard_reason'] = np.nan
                else:
                    qCtrl['keep_trial']     = np.nan
                    qCtrl['good_fit']       = np.nan
                    qCtrl['discard_reason'] = np.nan

                # save trial's fit data to a dataframe
                paramsSub   = pd.concat([paramsSub, pd.DataFrame([newResult], columns=newResult.keys())], ignore_index=True)
                qualityCtrl = pd.concat([qualityCtrl, pd.DataFrame([qCtrl], columns=qCtrl.keys())], ignore_index=True)

        nanOnsetpdf.close()
        nanOverallpdf.close()
        nanSequencepdf.close()

        pdf.close()
        plt.close('all')

        paramsSub.to_hdf(h5_file, 'data/')
        paramsRaw.to_hdf(h5_rawfile, 'raw/')
        qualityCtrl.to_hdf(h5_qcfile, 'data/')

        # test if it can read the file
        abc = pd.read_hdf(h5_file, 'data/')
        abc.head()

        del paramsRaw, abc, paramsSub, qualityCtrl, newResult