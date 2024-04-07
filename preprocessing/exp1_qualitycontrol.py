import os
import sys
import numpy as np
import pandas as pd

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
# manual_qc = pd.read_excel('qc_line.xlsx', sheet_name='qc_line') 
manual_qc = pd.read_csv('qc_line_noSPss.csv', sep=';') 
for sub in subjects:
    print('Subject:',sub)
                          
    for cond in conditions:
        print(cond)
        
        # get bad data
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_no-SPss.h5'.format(sub=sub, cond=cond)
        cq        = pd.read_hdf(h5_qcfile, 'data/')
            
        missingTrials = list(set(np.arange(1,nTrials[cond]+1,1)) - set(cq['trial']))
        
        for t in missingTrials:
            cq = cq.append(pd.DataFrame([[sub,cond,t,0,np.nan,'missing?']], 
                                        columns=['sub', 'condition', 'trial', 'keep_trial', 'good_fit', 'discard_reason']))
        
        idx = manual_qc[(manual_qc['condition']==cond) & (manual_qc['sub']==sub)]['trial']
        cq.loc[cq['trial'].isin(idx),'keep_trial'] = 0
        cq.reset_index(inplace=True)
        
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl_no-SPss_final.h5'.format(sub=sub, cond=cond)
        cq.to_hdf(h5_qcfile, 'data/')