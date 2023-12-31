import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# data_dir
main_dir = "../data/biasAccelerationControl" 
os.chdir(main_dir)


subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']
conditions =   [
                'Va-100_V0-0', 
                'Vd-100_V0-0',
                'V1-100_V0-0', 
                'V2-100_V0-0',
                'V3-100_V0-0',
                'Va-75_Vd-25',
                'Vd-75_Va-25',
                ]


nTrials = {
    'Va-100_V0-0':80, 
    'Vd-100_V0-0':80, 
    'V1-100_V0-0':80, 
    'V2-100_V0-0':80, 
    'V3-100_V0-0':80, 
    'Va-75_Vd-25':180, 
    'Vd-75_Va-25':180}
    
for sub in subjects:
    print('Subject:',sub)
    
    # get manual quality control data
    manual_qc = pd.read_excel('qc_line.xlsx', sheet_name='{sub}'.format(sub=sub[-2:])) 
    
    for cond in conditions:
        print(cond)
        
        # get bad data
        h5_qcfile = '{sub}/{sub}_{cond}_qualityControl.h5'.format(sub=sub, cond=cond)
        cq        = pd.read_hdf(h5_qcfile, 'data/')
        
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