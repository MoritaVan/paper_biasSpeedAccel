import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'


os.chdir('../functions')
from functions.utils import *

import warnings
warnings.filterwarnings("ignore")

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

equation = 'line'

#%% Parameters
# run always
screen_width_px  = 1920 # px
screen_height_px = 1080 # px
screen_width_cm  = 70   # cm
viewingDistance  = 57.  # cm

tan              = np.arctan((screen_width_cm/2)/viewingDistance)
screen_width_deg = 2. * tan * 180/np.pi
px_per_deg       = screen_width_px / screen_width_deg

def switch_X(argument):
    switcher = {
        'DR': 1,
        'UR': 1,
        'DL': -1,
        'UL': -1
    }
    return switcher.get(argument)

def switch_Y(argument):
    switcher = {
        'DR': -1,
        'UR': 1,
        'DL': -1,
        'UL': 1
    }
    return switcher.get(argument)
    
for sub in subjects:
    print('Subject:',sub)
    
    # get manual quality control data
    manual_qc = pd.read_excel('qc_line.xlsx', sheet_name='{sub}'.format(sub=sub[-2:])) 
    
    for cond in conditions:
        print(cond)
        
        # get bad data
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl.h5'.format(sub=sub, cond=cond)
        cq        = pd.read_hdf(h5_qcfile, 'data/')
        
        if sub == 'sub-13':
            missingTrials = list(set(np.arange(0,nTrialsSub13[cond],1)) - set(cq['trial']))
        else:
            missingTrials = list(set(np.arange(0,nTrials[cond],1)) - set(cq['trial']))
        
        for t in missingTrials:
            cq = cq.append(pd.DataFrame([[sub,cond,t,0,np.nan,'missing?']], 
                                        columns=['sub', 'condition', 'trial', 'keep_trial', 'good_fit', 'discard_reason']))
        
        idx = manual_qc[(manual_qc['cond']==cond) & (manual_qc['keep_trial']==0)]['trial']
        cq.loc[cq['trial'].isin(idx),'keep_trial'] = 0
        idx = manual_qc[(manual_qc['cond']==cond) & (manual_qc['bad_fit']==1)]['trial']
        cq.loc[cq['trial'].isin(idx),'good_fit'] = 0

        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_afterManualCtrl.h5'.format(sub=sub, cond=cond)
        cq.to_hdf(h5_qcfile, 'data/')