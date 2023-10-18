#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  30 12:22:01 2020

@author: Vanessa Morita


"""
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("../functions")
from functions.utils import *
from functions.updateRC import fontsizeDict,rcConfigDict

plt.rcParams.update(fontsizeDict(small=10,medium=12))
plt.rcParams.update(rcConfigDict(filepath = "../rcparams_config.json"))

cm = 1/2.54  # centimeters in inches
single_col = 9*cm
# oneDot5_col = 12.7*cm
two_col = 19*cm

main_dir = "../data/biasSpeed"
os.chdir(main_dir)

import warnings
warnings.filterwarnings("ignore")
import traceback

output_folder = '../outputs/exp1'
#%% Parameters
subjects   = ['s1', 's2', 's3']
subNum     = [int(x[-1]) for x in subjects]

colormaps = [ 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            ]
colormaps = {sub : col for sub,col in zip(subNum,colormaps)}

conditions =   [
                'p0', 
                'p10',
                'p25', 
                'p50',
                'p75',
                'p90',
                'p100',
                ]

cond_num = np.array([
    0, 10, 25, 50, 75, 90 , 100
])/100
print(cond_num)

# read data
print("Reading Data")
for sub in subjects:
    h5_file = ''.join([sub,'/', sub, '_biasSpeed_smoothPursuitData.h5'])
    data_tmp =  pd.read_hdf(h5_file, 'data')
    data_tmp.reset_index(inplace=True)
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-1])
    data_tmp['sub_txt'] = sub
    data_tmp['cond_num'] = [float(x[1:])/100 for x in data_tmp['cond']]
    
    data_tmp['ramp_pursuit_x'][data_tmp['ramp_pursuit_x'] > 250]  = np.nan
    data_tmp['ramp_pursuit_x'][data_tmp['ramp_pursuit_x'] < -250] = np.nan
    data_tmp['steady_state_x'][data_tmp['steady_state_x'] > 20] = np.nan
    data_tmp['steady_state_x'][data_tmp['steady_state_x'] < 0]  = np.nan
    
    data_tmp['vel_anti_x'] = data_tmp['velocity_model_x']
    data_tmp['vel_anti_x'][data_tmp['vel_anti_x']==0] = np.nan

    data_tmp['start_anti_x'][data_tmp['latency_x'] == data_tmp['start_anti_x']+1] = np.nan
    
    data_tmp['rotVel_x'] = data_tmp['velocity_model_x']

    # reading the data from the two regressions
    for cond in conditions:
        h5_file2 = '{sub}/{sub}_{cond}_posFilter_classical2.h5'.format(sub=sub, cond=cond) 
        data_tmp2 =  pd.read_hdf(h5_file2, 'data')
        data_tmp2.reset_index(inplace=True)
        
        if cond =='p50': # data files from p50 had wrong names
            new_val = {'HS': "LS", "LS": "HS"}
            data_tmp2.replace({"trial_vel": new_val}, inplace=True)

        idxcond = data_tmp[(data_tmp['cond']==cond)].index
        idxvel  = data_tmp2[(data_tmp2['trial'].isin(data_tmp.loc[idxcond,'trial']))].index

        data_tmp.loc[idxcond,'latency_x']      = list(data_tmp2.loc[idxvel, 'latency'])
        data_tmp.loc[idxcond,'ramp_pursuit_x'] = list(data_tmp2.loc[idxvel, 'a_pur']*1000)

    if sub == subjects[0]:
        dataSub = data_tmp.groupby(['cond']).mean()
        allSubsData = data_tmp
    else:
        tmp = data_tmp.groupby(['cond']).mean()
        dataSub = dataSub.append(tmp)
        allSubsData = allSubsData.append(data_tmp)

# read LME results from csv generated on R
lmm_dir = "{}/exp1/LMM".format(output_folder)
lme_raneff     = pd.read_csv('{}/lme_biasSpeed_randomEffects2.csv'.format(lmm_dir))
lme_fixeffAnti = pd.read_csv('{}/lme_biasSpeed_fixedeffectsAnti.csv'.format(lmm_dir))
lme_fixeffVGP  = pd.read_csv('{}/lme_biasSpeed_fixedeffectsVGP2.csv'.format(lmm_dir))

lme_fixeffAnti.at[0,'Unnamed: 0'] = 'Intercept'
lme_fixeffVGP.at[0,'Unnamed: 0']  = 'Intercept'
lme_fixeffAnti.set_index('Unnamed: 0', inplace=True)
lme_fixeffVGP.set_index('Unnamed: 0', inplace=True)

lme_fixeffAnti.fillna(0, inplace=True)
lme_fixeffVGP.fillna(0, inplace=True)
lme_raneff.fillna(0, inplace=True)

anticipParams = [['start_anti_x','Anticipation Onset', [-130,-40], 'Horizontal aSPon (ms)'],
                 ['velocity_model_x', 'Anticipatory Eye Velocity', [-2.5,9], 'Horizontal aSPv (°/s)'],
                ]
anticipData = allSubsData.groupby(['sub','cond_num']).mean()
anticipData.reset_index(inplace=True)

print(lme_fixeffAnti)
print(lme_fixeffVGP)
print(lme_raneff)

for p in anticipParams:
    print(p[1])

    intercept = lme_fixeffAnti.loc['Intercept',p[0]]
    prob      = lme_fixeffAnti.loc['prob',p[0]]

    
    # fig1 = plt.figure(figsize=(single_col*0.5, 3*cm)) 
    f,ax1 = plt.subplots(figsize=(single_col, 6*cm))# width, height
    plt.suptitle(p[1])
    ax1 = plt.subplot(1,1,1)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-1:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        
        reg = cond_num*(prob+s_prob) + (intercept + s_intercept)
        ax1.plot(cond_num, reg, color=np.array([200,200,200])/255)
    plotBoxDispersion(data=anticipData, 
                        by='cond_num', 
                        between=p[0], cmapAlpha=1,
                        scatterSize=25,
                        jitter=.01,
                        xticks= cond_num,
                        boxWidth = .055,
                        showfliers=False,
                        showKde=False,
                        cmapName='Blues_r',
                        ax=ax2)
    plt.ylim(p[2])
    ax1.set_xlabel('P(HS)')
    ax1.set_ylabel(p[3])
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])

    plt.tight_layout()
    plt.savefig('{}/biasSpeed_group_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/biasSpeed_group_{}.png'.format(output_folder, p[1]))

# plt.close('all')   
# plt.figure()
# plt.subplot(1,2,1)
# sns.boxplot(data=allSubsData, x='cond_num', y='latency_x', hue='trial_vel')
# plt.subplot(1,2,2)
# sns.boxplot(data=allSubsData, x='cond_num', y='ramp_pursuit_x', hue='trial_vel')
# plt.show()

visParams = [['latency_x','Latency', [75,125], 'Horizontal SPlat (ms)'],
             ['ramp_pursuit_x', 'Pursuit Acceleration', [35,150], 'Horizontal SPacc (°/s\N{SUPERSCRIPT TWO})'],
             ['steady_state_x', 'Steady State', [4, 17], 'Horizontal SPss (°/s)'],
            ]
visData = allSubsData.groupby(['sub','cond_num','trial_vel']).agg(np.nanmean)
visData.reset_index(inplace=True)

print(visData)

idxHS   = [True if 'H' in x else False for x in visData['trial_vel']]
idxLS = [not x for x in idxHS]

idxHS = np.array(idxHS)
idxLS = np.array(idxLS)

for p in visParams:
    print(p[1])

    intercept  = lme_fixeffVGP.loc['Intercept',p[0]]
    prob       = lme_fixeffVGP.loc['prob',p[0]]
    tgvel      = lme_fixeffVGP.loc['tgVelLS',p[0]]
    prob_tgvel = lme_fixeffVGP.loc['prob:tgVelLS',p[0]]
    
    fig1 = plt.subplots(figsize=(single_col, 6*cm)) # width, height
    plt.suptitle(p[1])
    ax1 = plt.subplot(1,1,1) 
    ax2 = ax1.twiny() 
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-1]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        s_tgvel     = list(raneff['tgVel'])[0]
        
        regHS = (prob+s_prob)*cond_num[1:] + (tgvel+s_tgvel)*0 + prob_tgvel*0*cond_num[1:] + (intercept+s_intercept)
        regLS = (prob+s_prob)*cond_num[:-1] + (tgvel+s_tgvel)*1 + prob_tgvel*1*cond_num[:-1] + (intercept+s_intercept)
        ax1.plot(cond_num[1:]+.03, regHS, color=np.array([200,220,255])/255, alpha=0.7)
        ax1.plot(cond_num[:-1]-.03, regLS, color=np.array([150,255,150])/255, alpha=0.7)
    plotBoxDispersion(data=visData[idxHS], 
                        by=['cond_num'], 
                        between=p[0], ax=ax2, alpha=0,
                        cmapAlpha=1,
                        scatterSize=25,
                        jitter=.01,
                        xticks= cond_num[1:]+.03,
                        boxWidth = .045,
                        showfliers=False,
                        showKde=False,
                        cmapName='Blues_r')
    plotBoxDispersion(data=visData[idxLS], 
                        by=['cond_num'], 
                        between=p[0], ax=ax2, alpha=0,  
                        cmapAlpha=1,
                        boxWidth=.045,
                        xticks=cond_num[:-1]-.03,
                        scatterSize=25,
                        jitter=.01,
                        showfliers=False,
                        showKde=False,
                        cmapName='Greens_r')
    
    # plt.ylim(p[2])
    ax1.set_xlabel('P(HS)')
    ax1.set_ylabel(p[3])
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])
    # ax.set_xlabels(conditions)
    # plt.xticks(rotation = 20)
    ax1.legend(['HS', 'LS'])
   
    plt.tight_layout()
    
    plt.savefig('{}/biasSpeed_group_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/biasSpeed_group_{}.png'.format(output_folder, p[1]))