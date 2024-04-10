import os
import sys
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

sys.path.append('./')

from functions.utils import *

import warnings
warnings.filterwarnings("ignore")
import traceback

# data_dir
main_dir = "../data/biasAcceleration" 
os.chdir(main_dir)


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

nTrials = {
    'Va-100_V0-0':100, 
    'Vd-100_V0-0':100, 
    'V1-100_V0-0':100, 
    'V2-100_V0-0':100, 
    'V3-100_V0-0':100, 
    'V1-75_V3-25':200, 
    'V3-75_V1-25':200, 
    'Va-75_Vd-25':200, 
    'Vd-75_Va-25':200}

nTrialsSub13 = {
    'Va-100_V0-0':80, 
    'Vd-100_V0-0':80, 
    'V1-100_V0-0':80, 
    'V2-100_V0-0':80, 
    'V3-100_V0-0':80,
    'Va-75_Vd-25':180, 
    'Vd-75_Va-25':180}

t1 = np.linspace(0,1,1000)
v1 = 11*np.ones(len(t1)) / np.sqrt(2)
t2 = np.linspace(0,0.82,820)
v2 = 22*np.ones(len(t2)) / np.sqrt(2)
t3 = np.linspace(0,0.52,520)
v3 = 33*np.ones(len(t3)) / np.sqrt(2)
ta = np.linspace(0,0.87,870)
va = (11 + 22*ta) / np.sqrt(2)
td = np.linspace(0,0.72,720)
vd = (33 - 22*td) / np.sqrt(2)

velocities = {
    'V1': {
        't': t1,
        'v': v1,
    },
    'V2': {
        't': t2,
        'v': v2,
    },
    'V3': {
        't': t3,
        'v': v3,
    },
    'Va': {
        't': ta,
        'v': va,
    },
    'Vd': {
        't': td,
        'v': vd,
    },
}

#%% Parameters
# run always
screen_width_px  = 1920 # px
screen_height_px = 1080 # px
screen_width_cm  = 70   # cm
viewingDistance  = 57.  # cm

tan              = np.arctan((screen_width_cm/2)/viewingDistance)
screen_width_deg = 2. * tan * 180/np.pi
px_per_deg       = screen_width_px / screen_width_deg
    
# for sub in subjects:
#     print('Subject:',sub)
    
#     # get manual quality control data
#     manual_qc = pd.read_excel('qc_line.xlsx', sheet_name='{sub}'.format(sub=sub[-2:])) 
    
#     for cond in conditions:
#         print(cond)
        
#         # get bad data
#         h5_qcfile = '{sub}/{sub}_{cond}_qualityControl.h5'.format(sub=sub, cond=cond)
#         cq        = pd.read_hdf(h5_qcfile, 'data/')
        
#         if sub == 'sub-13':
#             missingTrials = list(set(np.arange(0,nTrialsSub13[cond],1)) - set(cq['trial']))
#         else:
#             missingTrials = list(set(np.arange(0,nTrials[cond],1)) - set(cq['trial']))
        
#         for t in missingTrials:
#             cq = cq.append(pd.DataFrame([[sub,cond,t,0,np.nan,'missing?']], 
#                                         columns=['sub', 'condition', 'trial', 'keep_trial', 'good_fit', 'discard_reason']))
        
#         idx = manual_qc[(manual_qc['cond']==cond) & (manual_qc['keep_trial']==0)]['trial']
#         cq.loc[cq['trial'].isin(idx),'keep_trial'] = 0
#         idx = manual_qc[(manual_qc['cond']==cond) & (manual_qc['bad_fit']==1)]['trial']
#         cq.loc[cq['trial'].isin(idx),'good_fit'] = 0

#         h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_afterManualCtrl.h5'.format(sub=sub, cond=cond)
#         cq.to_hdf(h5_qcfile, 'data/')

for sub in subjects:
    print('Subject:',sub)
    outputFolder_plots = '{sub}/plots/'.format(sub=sub)
                          
    for cond in conditions:
        print(cond)
        try:
            h5_file   = '{sub}/{sub}_{cond}_posFilter_nonlinear.h5'.format(sub=sub, cond=cond) 
            h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(sub=sub, cond=cond)
            badFitsFile = '{}/qc/{}_{}_badFits_nonlinear.pdf'.format(outputFolder_plots, sub, cond)   
            data      = pd.read_hdf(h5_file, 'data/') 
            cq        = pd.read_hdf(h5_qcfile, 'data/')
        except:
            print('Error! \n Couldn\'t process {}, condition {}'.format(sub,cond))
            traceback.print_exc()

        badFitspdf      = PdfPages(badFitsFile)

        f = plt.figure()
        badFitspdf.savefig(f)

        data.dropna(how='all', axis=0,subset=['trial_velocity'], inplace=True)

        cq.reset_index(inplace=True)
        cq_index = cq[cq['discard_reason'] == 'RMSE criteria'].index
        cq.loc[cq_index,'good_fit']       = [np.nan] * len(cq_index)
        cq.loc[cq_index,'discard_reason'] = [np.nan] * len(cq_index)
        

        for idx, row in data.iterrows():
            idx_notnan = ~np.isnan(row['velocity_x'])
            time_noNan = np.array(row.time_x[idx_notnan])
            idx_res = (time_noNan>-200)&(time_noNan<200)
            data.at[idx,'rmse_x_-200-200ms'] = np.sqrt(np.mean([x*x for x in row.residual_x[idx_res]]))

            idx_notnan = ~np.isnan(row['velocity_y'])
            time_noNan = np.array(row.time_y[idx_notnan])
            idx_res = (time_noNan>-200)&(time_noNan<200)
            data.at[idx,'rmse_y_-200-200ms'] = np.sqrt(np.mean([x*x for x in row.residual_y[idx_res]]))

        rmse_x_threshold = data['rmse_x'].mean() + 2*data['rmse_x'].std()
        rmse_y_threshold = data['rmse_y'].mean() + 2*data['rmse_y'].std()
        rmse_x_threshold_fixWin = data['rmse_x_-200-200ms'].mean() + 2*data['rmse_x_-200-200ms'].std()
        rmse_y_threshold_fixWin = data['rmse_y_-200-200ms'].mean() + 2*data['rmse_y_-200-200ms'].std()
        print('RMSE_x thresh: ', rmse_x_threshold, rmse_x_threshold_fixWin)
        print('RMSE_y thresh: ', rmse_y_threshold, rmse_y_threshold_fixWin)

        for idx, row in data.iterrows():

            if (row['rmse_x_-200-200ms'] > rmse_x_threshold_fixWin) | (row.rmse_x > rmse_x_threshold) | (row['rmse_y_-200-200ms'] > rmse_y_threshold_fixWin) | (row.rmse_y > rmse_y_threshold):#| (row.rmse_x > 1.5) | (row['rmse_-200-200ms'] > 1.5):
                print(row.trial)

                target_time = velocities[row.trial_velocity]['t']*1000
                target_vel  = velocities[row.trial_velocity]['v']
                f = plotFig2(row.trial, target_time, target_vel, 'U', 'R', # type_v, type_h,
                                        row['time_y'], row['velocity_y'], row['fit_y'], row['aSPon_y'], row['aSPoff_y'],
                                        row['time_x'], row['velocity_x'], row['fit_x'], row['aSPon_x'], row['aSPoff_x'],
                                        show=False)
                badFitspdf.savefig(f)
                plt.close(f)

                cq_index = cq[cq.trial==row.trial].index[0]
                cq.at[cq_index,'good_fit'] = 0
                cq.at[cq_index,'discard_reason'] = 'RMSE criteria'


        missingTrials = list(set(np.arange(1,nTrials[cond]+1,1)) - set(cq['trial']))
        
        for t in missingTrials:
            cq = cq.append(pd.DataFrame([[sub,cond,t,0,np.nan,'missing?']], 
                                        columns=['sub', 'condition', 'trial', 'keep_trial', 'good_fit', 'discard_reason']))
        
        badFitspdf.close()
        
        cq.to_hdf(h5_qcfile, 'data/')