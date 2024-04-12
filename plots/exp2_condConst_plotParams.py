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
two_col = 19*cm

main_dir = "../data/biasAcceleration"
os.chdir(main_dir)

import warnings
warnings.filterwarnings("ignore")
import traceback

output_folder = '../outputs/exp2'

subjects   = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06',
                'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13'] 

conditions =   [
                'V1-100_V0-0', 
                'V3-100_V0-0',
                'V1-75_V3-25',
                'V3-75_V1-25',
                ]

condCte = {
                'V1-100_V0-0': .0, 
                'V1-75_V3-25': .30,
                'V3-75_V1-25': .70,
                'V3-100_V0-0': 1.00,
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
    
    h5_file = ''.join([sub,'/', sub, '_biasAccel_smoothPursuitData_nonlinear.h5'])
    data_tmp =  pd.read_hdf(h5_file, 'data')
    data_tmp.reset_index(inplace=True)

    data_tmp = data_tmp[data_tmp['condition'].isin(conditions)]

    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-2:])
    data_tmp['sub_txt'] = sub
    data_tmp['prob'] = [condCte[x] for x in data_tmp['condition']]
    
    data_tmp.loc[data_tmp['SPlat_x'] == data_tmp['aSPon_x']+1, 'aSPon_x'] = np.nan
    data_tmp.loc[data_tmp['SPlat_y'] == data_tmp['aSPon_y']+1, 'aSPon_y'] = np.nan

    for cond in conditions:
        try:
            n1_tgVel     = list(data_tmp.loc[data_tmp['condition']==cond,'trial_velocity'])
            n1_tgVel[1:] = n1_tgVel[:-1]
            n1_tgVel[0]  = np.nan
            data_tmp.loc[data_tmp['condition']==cond,'n1_tgVel'] = n1_tgVel
        except:
            print('Cond not found')

        allSubsData = pd.concat([allSubsData,data_tmp], ignore_index=True)

allSubsData = allSubsData[allSubsData['trial']>10]
new_val = {'V1': "v11", "V2": "v22", "V3":"v33", "Va":"vacc", "Vd":"vdec"}
allSubsData.replace({"trial_velocity": new_val}, inplace=True)
allSubsData.replace({"n1_tgVel": new_val}, inplace=True)
allSubsData.reset_index(inplace=True)
print(allSubsData.head())

anticipData = allSubsData.groupby(['n1_tgVel','sub','prob']).mean()
anticipData.reset_index(inplace=True)
anticipData = anticipData.loc[~anticipData.prob.isin([0.0, 1.0]),['n1_tgVel','sub','prob','aSPv_x','aSPv_y']]


diff_x = np.array(anticipData.loc[anticipData.n1_tgVel=='v33','aSPv_x'])-np.array(anticipData.loc[anticipData.n1_tgVel=='v11','aSPv_x'])
diff_y = np.array(anticipData.loc[anticipData.n1_tgVel=='v33','aSPv_y'])-np.array(anticipData.loc[anticipData.n1_tgVel=='v11','aSPv_y'])
diff_df = anticipData.loc[anticipData.n1_tgVel=='v33',['sub','prob']]
diff_df['diff_v33-v11_x'] = diff_x
diff_df['diff_v33-v11_y'] = diff_y
print(diff_df)
f = plt.figure(figsize=(two_col,6*cm))
plt.subplot(1,2,1)
sns.lineplot(data=diff_df,
             x = 'prob', y = 'diff_v33-v11_x',
             err_style="bars", markers=True)
plt.ylim([0,1.3])
plt.xlim([0,1])
plt.subplot(1,2,2)
sns.lineplot(data=diff_df,
             x = 'prob', y = 'diff_v33-v11_y',
             err_style="bars", markers=True)
plt.ylim([0,1.3])
plt.xlim([0,1])
plt.savefig('{}/exp2_diff_v33-v11.png'.format(output_folder))
plt.savefig('{}/exp2_diff_v33-v11.pdf'.format(output_folder))

dhasj

# read LME results from csv generated on R
lmm_dir = "{}/LMM".format(output_folder)
lme_raneff     = pd.read_csv('{}/exp2_condConst_lmm_randomEffects.csv'.format(lmm_dir))
lme_fixeffAnti = pd.read_csv('{}/exp2_condConst_lmm_fixedeffectsAnti.csv'.format(lmm_dir))
lme_fixeffVGP  = pd.read_csv('{}/exp2_condConst_lmm_fixedeffectsVGP.csv'.format(lmm_dir))

lme_fixeffAnti.at[0,'Unnamed: 0'] = 'Intercept'
lme_fixeffVGP.at[0,'Unnamed: 0']  = 'Intercept'
lme_fixeffAnti.set_index('Unnamed: 0', inplace=True)
lme_fixeffVGP.set_index('Unnamed: 0', inplace=True)

lme_fixeffAnti.fillna(0, inplace=True)
lme_fixeffVGP.fillna(0, inplace=True)
lme_raneff.fillna(0, inplace=True)

anticipParams = [['aSPon','Anticipation Onset', [-200,0], ' aSPon (ms)'],
                 ['aSPv', 'Anticipatory Eye Velocity', [-2.5,12], ' aSPv (°/s)'],
                ]
anticipData = allSubsData.groupby(['sub','prob','n1_tgVel']).mean()
anticipData.reset_index(inplace=True)
print(anticipData)
print(lme_fixeffAnti)
print(lme_fixeffVGP)
print(lme_raneff)

idxHS   = [True if '33' in x else False for x in anticipData['n1_tgVel']]
idxLS = [not x for x in idxHS]

idxHS = np.array(idxHS)
idxLS = np.array(idxLS)

for p in anticipParams:
    print(p[1])

    intercept = lme_fixeffAnti.loc['Intercept',p[0]]
    prob      = lme_fixeffAnti.loc['prob',p[0]]
    n1_tgVel  = lme_fixeffAnti.loc['n1_velV1',p[0]]
    axis      = lme_fixeffAnti.loc['axisvert.',p[0]]
    prob_n1   = lme_fixeffAnti.loc['n1_velV1:prob',p[0]]
    prob_axis = lme_fixeffAnti.loc['prob:axisvert.',p[0]]
    axis_n1   = lme_fixeffAnti.loc['n1_velV1:axisvert.',p[0]]
    prob_n1   = lme_fixeffAnti.loc['n1_velV1:prob',p[0]]
    prob_n1_axis = lme_fixeffAnti.loc['n1_velV1:prob:axisvert.',p[0]]
    
    f,ax1 = plt.subplots(figsize=(two_col, 6*cm))# width, height
    plt.suptitle(p[1])

    # X AXIS
    var2plot = '{}_x'.format(p[0])
    y_label = 'Horizontal {}'.format(p[3])
    ax1 = plt.subplot(1,2,1)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-2:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        s_n1Vel     = list(raneff['n1_tgVel'])[0]
        s_axis      = list(raneff['axis'])[0]
        
        reg_n1HS = cond_num[1:]*(prob + s_prob) + (intercept + s_intercept)
        reg_n1LS = cond_num[:-1]*(prob_n1 + prob + s_prob) + (s_n1Vel + n1_tgVel) + (intercept + s_intercept)
        ax1.plot(cond_num[1:]+.03, reg_n1HS, color=colorHS, alpha=0.5)
        ax1.plot(cond_num[:-1]-.03, reg_n1LS, color=colorLS, alpha=0.5)
    
    plotBoxDispersion(data=anticipData[idxHS], 
                        by=['prob'], 
                        between=var2plot, ax=ax2, alpha=0,
                        cmapAlpha=1,
                        scatterSize=25,
                        jitter=.01,
                        xticks= cond_num[1:]+.03,
                        boxWidth = .045,
                        showfliers=False,
                        showKde=False,
                        color = colorHS,
                        cmapName=None)
    plotBoxDispersion(data=anticipData[idxLS], 
                        by=['prob'], 
                        between=var2plot, ax=ax2, alpha=0,  
                        cmapAlpha=1,
                        boxWidth=.045,
                        xticks=cond_num[:-1]-.03,
                        scatterSize=25,
                        jitter=.01,
                        showfliers=False,
                        showKde=False,
                        color = colorLS,
                        cmapName=None)
    plt.ylim(p[2])
    ax1.set_xlabel('P(v33)')
    ax1.set_ylabel(y_label)
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])

    # Y AXIS
    var2plot = '{}_y'.format(p[0])
    y_label = 'Vertical {}'.format(p[3])
    ax1 = plt.subplot(1,2,2)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-2:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        s_n1Vel     = list(raneff['n1_tgVel'])[0]
        s_axis      = list(raneff['axis'])[0]
        
        reg_n1HS = cond_num[1:]*(prob_axis + prob + s_prob) + (axis + s_axis) + (intercept + s_intercept)
        reg_n1LS = cond_num[:-1]*(prob_n1 + prob_axis + prob_n1_axis + prob + s_prob) + (s_n1Vel + n1_tgVel) + (axis + axis_n1 + s_axis) + (intercept + s_intercept)
        ax1.plot(cond_num[1:]+.03, reg_n1HS, color=colorHS, alpha=0.5)
        ax1.plot(cond_num[:-1]-.03, reg_n1LS, color=colorLS, alpha=0.5)
    
    plotBoxDispersion(data=anticipData[idxHS], 
                        by=['prob'], 
                        between=var2plot, ax=ax2, alpha=0,
                        cmapAlpha=1,
                        scatterSize=25,
                        jitter=.01,
                        xticks= cond_num[1:]+.03,
                        boxWidth = .045,
                        showfliers=False,
                        showKde=False,
                        color = colorHS,
                        cmapName=None)
    plotBoxDispersion(data=anticipData[idxLS], 
                        by=['prob'], 
                        between=var2plot, ax=ax2, alpha=0,  
                        cmapAlpha=1,
                        boxWidth=.045,
                        xticks=cond_num[:-1]-.03,
                        scatterSize=25,
                        jitter=.01,
                        showfliers=False,
                        showKde=False,
                        color = colorLS,
                        cmapName=None)
    plt.ylim(p[2])
    ax1.set_xlabel('P(v33)')
    ax1.set_ylabel(y_label)
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])
    ax1.legend(['N-1=v33', 'N-1=v11'])

    plt.tight_layout()
    plt.savefig('{}/exp2_condConst_group_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/exp2_condConst_group_{}.png'.format(output_folder, p[1]))

visParams = [['SPlat','Latency', [70,170], 'SPlat (ms)'],
             ['SPacc', 'Pursuit Acceleration', [20,320], 'SPacc (°/s\N{SUPERSCRIPT TWO})'],
            #  ['SPss', 'Steady State', [4, 17], 'Horizontal SPss (°/s)'],
            ]
visData = allSubsData.groupby(['sub','prob','trial_velocity']).agg(np.nanmean)
visData.reset_index(inplace=True)

idxHS   = [True if '33' in x else False for x in visData['trial_velocity']]
idxLS = [not x for x in idxHS]

idxHS = np.array(idxHS)
idxLS = np.array(idxLS)

for p in visParams:
    print(p[1])

    intercept  = lme_fixeffVGP.loc['Intercept',p[0]]
    prob       = lme_fixeffVGP.loc['prob',p[0]]
    tgvel      = lme_fixeffVGP.loc['trial_velocityV1',p[0]]
    axis       = lme_fixeffVGP.loc['axisvert.',p[0]]
    prob_tgvel = lme_fixeffVGP.loc['trial_velocityV1:prob',p[0]]
    prob_axis  = lme_fixeffVGP.loc['prob:axisvert.',p[0]]
    axis_tgvel = lme_fixeffVGP.loc['trial_velocityV1:axisvert.',p[0]]
    prob_tgvel = lme_fixeffVGP.loc['trial_velocityV1:prob',p[0]]
    prob_tgvel_axis = lme_fixeffVGP.loc['trial_velocityV1:prob:axisvert.',p[0]]
    
    fig1 = plt.subplots(figsize=(two_col, 6*cm)) # width, height
    plt.suptitle(p[1])
    # X AXIS
    var2plot = '{}_x'.format(p[0])
    y_label = 'Horizontal {}'.format(p[3])
    ax1 = plt.subplot(1,2,1)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-2:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        s_tgVel     = list(raneff['trial_velocity'])[0]
        s_axis      = list(raneff['axis'])[0]
        
        reg_HS = cond_num[1:]*(prob + s_prob) + (intercept + s_intercept)
        reg_LS = cond_num[:-1]*(prob_tgvel + prob + s_prob) + (s_tgVel + tgvel) + (intercept + s_intercept)
        ax1.plot(cond_num[1:]+.03, reg_HS, color=colorHS, alpha=0.5)
        ax1.plot(cond_num[:-1]-.03, reg_LS, color=colorLS, alpha=0.5)
    
    plotBoxDispersion(data=visData[idxHS], 
                        by=['prob'], 
                        between=var2plot, ax=ax2, alpha=0,
                        cmapAlpha=1,
                        scatterSize=25,
                        jitter=.01,
                        xticks= cond_num[1:]+.03,
                        boxWidth = .045,
                        showfliers=False,
                        showKde=False,
                        color = colorHS,
                        cmapName=None)
    plotBoxDispersion(data=visData[idxLS], 
                        by=['prob'], 
                        between=var2plot, ax=ax2, alpha=0,  
                        cmapAlpha=1,
                        boxWidth=.045,
                        xticks=cond_num[:-1]-.03,
                        scatterSize=25,
                        jitter=.01,
                        showfliers=False,
                        showKde=False,
                        color = colorLS,
                        cmapName=None)
    plt.ylim(p[2])
    ax1.set_xlabel('P(v33)')
    ax1.set_ylabel(y_label)
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])

    # Y AXIS
    var2plot = '{}_y'.format(p[0])
    y_label = 'Vertical {}'.format(p[3])
    ax1 = plt.subplot(1,2,2)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-2:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        s_tgVel     = list(raneff['trial_velocity'])[0]
        s_axis      = list(raneff['axis'])[0]
        
        reg_HS = cond_num[1:]*(prob_axis + prob + s_prob) + (axis + s_axis) + (intercept + s_intercept)
        reg_LS = cond_num[:-1]*(prob_tgvel + prob_axis + prob_tgvel_axis + prob + s_prob) + (s_tgVel + tgvel) + (axis + axis_tgvel + s_axis) + (intercept + s_intercept)
        ax1.plot(cond_num[1:]+.03, reg_HS, color=colorHS, alpha=0.5)
        ax1.plot(cond_num[:-1]-.03, reg_LS, color=colorLS, alpha=0.5)
    
    plotBoxDispersion(data=visData[idxHS], 
                        by=['prob'], 
                        between=var2plot, ax=ax2, alpha=0,
                        cmapAlpha=1,
                        scatterSize=25,
                        jitter=.01,
                        xticks= cond_num[1:]+.03,
                        boxWidth = .045,
                        showfliers=False,
                        showKde=False,
                        color = colorHS,
                        cmapName=None)
    plotBoxDispersion(data=visData[idxLS], 
                        by=['prob'], 
                        between=var2plot, ax=ax2, alpha=0,  
                        cmapAlpha=1,
                        boxWidth=.045,
                        xticks=cond_num[:-1]-.03,
                        scatterSize=25,
                        jitter=.01,
                        showfliers=False,
                        showKde=False,
                        color = colorLS,
                        cmapName=None)
    plt.ylim(p[2])
    ax1.set_xlabel('P(v33)')
    ax1.set_ylabel(y_label)
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])
    # ax.set_xlabels(conditions)
    # plt.xticks(rotation = 20)
    ax1.legend(['v33', 'v11'])
   
    plt.tight_layout()
    
    plt.savefig('{}/exp2_condConst_group_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/exp2_condConst_group_{}.png'.format(output_folder, p[1]))