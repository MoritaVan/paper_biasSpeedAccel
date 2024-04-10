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

# data_dir
main_dir = "../data/biasSpeed" 
os.chdir(main_dir)


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

nTrials = {
    'p0':250, 
    'p10':500, 
    'p25':500, 
    'p50':400, 
    'p75':500, 
    'p90':500, 
    'p100':250, 
    }

# get manual quality control data
# manual_qc = pd.read_csv('qc_nonlinear.csv', sep=';') 
# for sub in subjects:
#     print('Subject:',sub)
                          
#     for cond in conditions:
#         print(cond)
        
#         # get bad data
#         h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(sub=sub, cond=cond)
#         cq        = pd.read_hdf(h5_qcfile, 'data/')
            
#         missingTrials = list(set(np.arange(1,nTrials[cond]+1,1)) - set(cq['trial']))
        
#         for t in missingTrials:
#             cq = cq.append(pd.DataFrame([[sub,cond,t,0,np.nan,'missing?']], 
#                                         columns=['sub', 'condition', 'trial', 'keep_trial', 'good_fit', 'discard_reason']))
        
#         idx = manual_qc[(manual_qc['condition']==cond) & (manual_qc['sub']==sub)]['trial']
#         cq.loc[cq['trial'].isin(idx),'keep_trial'] = 0
#         cq.reset_index(inplace=True)
        
#         h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_nonlinear_final.h5'.format(sub=sub, cond=cond)
#         cq.to_hdf(h5_qcfile, 'data/')

for sub in subjects:
    print('Subject:',sub)
    outputFolder_plots = '{sub}/plots/'.format(sub=sub)
                          
    for cond in conditions:
        print(cond)
        h5_file   = '{sub}/{sub}_{cond}_posFilter_nonlinear.h5'.format(sub=sub, cond=cond) 
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_nonlinear.h5'.format(sub=sub, cond=cond)
        badFitsFile = '{}/qc/{}_{}_badFits_nonlinear.pdf'.format(outputFolder_plots, sub, cond)   
        data      = pd.read_hdf(h5_file, 'data/') 
        cq        = pd.read_hdf(h5_qcfile, 'data/')

        badFitspdf      = PdfPages(badFitsFile)

        f = plt.figure()
        badFitspdf.savefig(f)

        for idx, row in data.iterrows():

            idx_notnan = ~np.isnan(row['velocity_x'])
            time_noNan = np.array(row.time[idx_notnan])
            idx_res = (time_noNan>-200)&(time_noNan<200)
            data.at[idx,'rmse_-200-200ms'] = np.sqrt(np.mean([x*x for x in row.residual_x[idx_res]]))

        rmse_threshold = data['rmse_x'].mean() + 1.5*data['rmse_x'].std()
        rmse_threshold_fixWin = data['rmse_-200-200ms'].mean() + 2*data['rmse_-200-200ms'].std()
        print('RMSE thresh: ', rmse_threshold, rmse_threshold_fixWin)

        for idx, row in data.iterrows():
            target_vel  = 5.5 if row.trial_velocity=='LS' else 16.5 

            if (row['rmse_-200-200ms'] > rmse_threshold_fixWin) | (row.rmse_x > rmse_threshold) | (row.rmse_x > 1.5) | (row['rmse_-200-200ms'] > 1.5):
                # print(row.trial)
                f = plotFig(row.trial, target_vel,
                        row['time'], row['velocity_x'], row['fit_x'], row['aSPon'], row['aSPoff'],
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
