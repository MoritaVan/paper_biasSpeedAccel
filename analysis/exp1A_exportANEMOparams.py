#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita


"""

import os
import numpy as np
import pandas as pd

main_dir = "../data/biasSpeed"
os.chdir(main_dir)

output_folder = '../outputs/exp1A'

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

# read data
print("Reading Data")

allSubsData = pd.DataFrame([])
for sub in subjects:
    h5_file = '{s}/{s}_biasSpeed_smoothPursuitData_nonlinear.h5'.format(s=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    data_tmp.reset_index(inplace=True)
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-1])
    data_tmp['sub_txt'] = sub
    data_tmp['cond_num'] = [float(x[1:])/100 for x in data_tmp['condition']]
    
    allSubsData = pd.concat([allSubsData,data_tmp],ignore_index=True)


print(allSubsData.shape)

# Linear Mixed effecs model - fit
# allData created above
allSubsData.to_csv('{}/exp1A_params.csv'.format(output_folder))
