#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita


"""
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('./')
from functions.utils import *
from functions.updateRC import fontsizeDict,rcConfigDict

plt.rcParams.update(fontsizeDict(small=10,medium=12))
plt.rcParams.update(rcConfigDict(filepath = "./rcparams_config.json"))

cm = 1/2.54  # centimeters in inches
single_col = 9*cm
# oneDot5_col = 12.7*cm
two_col = 17*cm

main_dir = "../data"
os.chdir(main_dir)

import warnings
warnings.filterwarnings("ignore")
import traceback

output_folder = "{}/outputs/exp2".format(main_dir)
data_dir_exp  = "{}/biasAcceleration".format(main_dir)
data_dir_ctrl = "{}/biasAccelerationControl".format(main_dir)
lmm_dir_exp   = "{}/outputs/exp2/LMM".format(main_dir)

subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06',
                'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13'] 
subjectsCtrl = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05']

newSubCtrl = {
    'sub-01': 'sub-01', 
    'sub-02': 'sub-13', 
    'sub-03': 'sub-16', 
    'sub-04': 'sub-17', 
    'sub-05': 'sub-18'
}

conditions =   [
                'Va-100_V0-0', 
                'Vd-100_V0-0',
                'Va-75_Vd-25',
                'Vd-75_Va-25',
                ]

condAcc = {
                'Va-100_V0-0': .0, 
                'Va-75_Vd-25': .30,
                'Vd-75_Va-25': .70,
                'Vd-100_V0-0': 1.00,
}
cond_num = np.array([
    0, 30, 70, 100
])/100
print(cond_num)

colorHS = np.array([185,53,64,255])/255
colorLS = np.array([64,103,199,255])/255

# read data
print("Reading Data")
allSubsData = pd.DataFrame([])
for sub in subjects:
    h5_file  = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_exp,sub=sub)
    data_tmp =  pd.read_hdf(h5_file, 'data')
    data_tmp = data_tmp[data_tmp['condition'].isin(conditions)]

    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-2:])
    data_tmp['sub_txt'] = sub
    data_tmp['prob'] = [condAcc[x] for x in data_tmp['condition']]
    
    data_tmp.loc[data_tmp['SPlat_x'] == data_tmp['aSPon_x']+1, 'aSPon_x'] = np.nan
    data_tmp.loc[data_tmp['SPlat_y'] == data_tmp['aSPon_y']+1, 'aSPon_y'] = np.nan

    allSubsData = pd.concat([allSubsData,data_tmp], ignore_index=True)

allSubsDataCtrl = pd.DataFrame([])
for sub in subjectsCtrl:
    try:
        h5_file = '{dir}/{sub}/{sub}_biasAccel_smoothPursuitData_nonlinear.h5'.format(dir=data_dir_ctrl,sub=sub)
        data_tmp =  pd.read_hdf(h5_file, 'data')
        data_tmp = data_tmp[data_tmp['condition'].isin(conditions)]

        data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-2:])
        data_tmp['sub_txt'] = sub
        data_tmp['prob'] = [condAcc[x] for x in data_tmp['condition']]
        
        data_tmp.loc[data_tmp['SPlat_x'] == data_tmp['aSPon_x']+1, 'aSPon_x'] = np.nan
        data_tmp.loc[data_tmp['SPlat_y'] == data_tmp['aSPon_y']+1, 'aSPon_y'] = np.nan

        allSubsDataCtrl = pd.concat([allSubsDataCtrl,data_tmp])

    except Exception as e:
        print('Error! \n Couldn\'t process {}'.format(sub))
        traceback.print_exc() 

allSubsData['exp']     = 'constDisplacement'
allSubsDataCtrl['exp'] = 'constDuration'
allSubsData     = allSubsData[allSubsData['trial']>10]
allSubsDataCtrl = allSubsDataCtrl[allSubsDataCtrl['trial']>10]
allSubsDataCtrl.replace({"sub": newSubCtrl}, inplace=True)
allSubsData = pd.concat([allSubsData,allSubsDataCtrl])

new_val = {'V1': "v11", "V2": "v22", "V3":"v33", "Va":"vacc", "Vd":"vdec"}
allSubsData.replace({"trial_velocity": new_val}, inplace=True)
allSubsData.replace({"n1_tgVel": new_val}, inplace=True)
allSubsData.reset_index(inplace=True)


# read LME results from csv generated on R
lmm_dir = "{}/LMM".format(output_folder)
lme_raneff     = pd.read_csv('{}/exp2_condAccel_lmm_randomEffects.csv'.format(lmm_dir))
lme_fixeffAnti = pd.read_csv('{}/exp2_condAccel_lmm_fixedeffectsAnti.csv'.format(lmm_dir))

lme_fixeffAnti.at[0,'Unnamed: 0'] = 'Intercept'
lme_fixeffAnti.set_index('Unnamed: 0', inplace=True)

lme_fixeffAnti.fillna(0, inplace=True)
lme_raneff.fillna(0, inplace=True)

anticipParams = [
                 ['aSPv', 'Anticipatory Eye Velocity', [-2.5,12], ' aSPv (°/s)'],
                ]
anticipData = allSubsData.groupby(['exp','sub','prob']).mean()
anticipData.reset_index(inplace=True)
print(anticipData.head())
print(lme_raneff)
print(lme_fixeffAnti)

cmapSpacing = [0.05, 0.3, 0.7, 0.95]

idxExp = [False if 'Duration' in x else True for x in anticipData['exp']]
idxCtrl = [not x for x in idxExp]

for p in anticipParams:
    print(p[1])

    intercept = lme_fixeffAnti.loc['Intercept',p[0]]
    prob      = lme_fixeffAnti.loc['prob',p[0]]
    axis      = lme_fixeffAnti.loc['axisvert.',p[0]]
    exp       = lme_fixeffAnti.loc['expconstantTime',p[0]]
    
    # prob_axis = lme_fixeffAnti.loc['prob:axisvert.',p[0]]
    # prob_exp  = lme_fixeffAnti.loc['prob:expconstantTime',p[0]]
    # axis_exp  = lme_fixeffAnti.loc['expconstantTime:axisvert.',p[0]]
    axis_exp  = lme_fixeffAnti.loc['axisvert.:expconstantTime',p[0]]

    # prob_ax_exp = lme_fixeffAnti.loc['prob:expconstantTime:axisvert.',p[0]]

    
    f,ax1 = plt.subplots(figsize=(two_col, 7*cm))# width, height
    plt.suptitle(p[1])

    #EXP 2A - const displacement
    ax1 = plt.subplot(1,2,1)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    plt.title('Exp 2A: Constant Displacement')
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-2:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        s_axis      = list(raneff['axis'])[0]
        
        regX = cond_num*(prob + s_prob) + (intercept + s_intercept)
        # regY = cond_num*(prob_axis + prob + s_prob) + (axis + s_axis) + (intercept + s_intercept)
        regY = cond_num*(prob + s_prob) + (axis + s_axis) + (intercept + s_intercept)
        ax1.plot(cond_num+.03, regX, color=colorHS, alpha=0.5)
        ax1.plot(cond_num-.03, regY, color=colorLS, alpha=0.5)
    
    plotBoxDispersion(data=anticipData[idxExp], 
                            by=['prob'], 
                            between='{}_x'.format(p[0]), ax=ax2, alpha=0,
                            cmapAlpha=1,
                            scatterSize=25,
                            jitter=.01,
                            xticks= cond_num+.03,
                            boxWidth = .045,
                            showfliers=False,
                            showKde=False,
                            color = colorHS,
                            cmapName=None)
    plotBoxDispersion(data=anticipData[idxExp], 
                            by=['prob'], 
                            between='{}_y'.format(p[0]), ax=ax2, alpha=0,  
                            cmapAlpha=1,
                            boxWidth=.045,
                            xticks=cond_num-.03,
                            scatterSize=25,
                            jitter=.01,
                            showfliers=False,
                            showKde=False,
                            color = colorLS,
                            cmapName=None)
    plt.ylim(p[2])
    ax1.set_xlabel('P(vdec)')
    ax1.set_ylabel(p[3])
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])

    # EXP 2B - const duration
    ax1 = plt.subplot(1,2,2)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    plt.title('Exp 2B: Constant Duration')
    for sub in list(newSubCtrl.values()):
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-2:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        s_axis      = list(raneff['axis'])[0]
        # s_exp       = list(raneff['experiment'])[0]

        # regX = cond_num*(prob + s_prob) + (exp + s_exp) + (intercept + s_intercept)
        regX = cond_num*(prob + s_prob) + exp + (intercept + s_intercept)
        # regY = cond_num*(prob_ax_exp + prob_axis + prob + s_prob) + (axis_exp + axis + s_axis) + (exp + s_exp) + (intercept + s_intercept)
        regY = cond_num*(prob + s_prob) + (axis_exp + axis + s_axis) + exp + (intercept + s_intercept)
        ax1.plot(cond_num+.03, regX, color=colorHS, alpha=0.5)
        ax1.plot(cond_num-.03, regY, color=colorLS, alpha=0.5)
        
    plotBoxDispersion(data=anticipData[idxCtrl], 
                            by=['prob'], 
                            between='{}_x'.format(p[0]), ax=ax2, alpha=0,
                            cmapAlpha=1,
                            scatterSize=25,
                            jitter=.01,
                            xticks= cond_num+.03,
                            boxWidth = .045,
                            showfliers=False,
                            showKde=False,
                            color = colorHS,
                            cmapName=None)
    plotBoxDispersion(data=anticipData[idxCtrl], 
                            by=['prob'], 
                            between='{}_y'.format(p[0]), ax=ax2, alpha=0,  
                            cmapAlpha=1,
                            boxWidth=.045,
                            xticks=cond_num-.03,
                            scatterSize=25,
                            jitter=.01,
                            showfliers=False,
                            showKde=False,
                            color = colorLS,
                            cmapName=None)
    plt.ylim(p[2])
    ax1.set_xlabel('P(vdec)')
    ax1.set_ylabel(p[3])
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])
    ax1.legend(['Horizontal', 'Vertical'])

    plt.tight_layout()
    plt.savefig('{}/exp2_condAccel_group_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/exp2_condAccel_group_{}.png'.format(output_folder, p[1]))
