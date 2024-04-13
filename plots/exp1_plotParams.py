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

main_dir = "../data/biasSpeed"
os.chdir(main_dir)

import warnings
warnings.filterwarnings("ignore")
import traceback

output_folder = '../outputs/exp1'
#%% Parameters
subjects   = ['s1', 's2', 's3']
subNum     = [int(x[-1]) for x in subjects]

# colormaps = [ 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             ]
# colormaps = {sub : col for sub,col in zip(subNum,colormaps)}

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

colorHS = np.array([185,53,64,255])/255
colorLS = np.array([64,103,199,255])/255

# read data
print("Reading Data")
allSubsData = pd.DataFrame([])
for sub in subjects:
    h5_file = ''.join([sub,'/', sub, '_biasSpeed_smoothPursuitData_nonlinear.h5'])
    data_tmp =  pd.read_hdf(h5_file, 'data')
    data_tmp.reset_index(inplace=True)
    
    data_tmp['sub'] = np.ones(len(data_tmp)) * int(sub[-1])
    data_tmp['sub_txt'] = sub
    data_tmp['cond_num'] = [float(x[1:])/100 for x in data_tmp['condition']]
    
    data_tmp.loc[data_tmp['SPlat'] == data_tmp['aSPon']+1, 'aSPon'] = np.nan

    for cond in conditions:
        n1_tgVel     = list(data_tmp.loc[data_tmp['condition']==cond,'trial_velocity'])
        n1_tgVel[1:] = n1_tgVel[:-1]
        n1_tgVel[0]  = np.nan
        data_tmp.loc[data_tmp['condition']==cond,'n1_tgVel'] = n1_tgVel
        

    # # reading the data from the two regressions
    # for cond in conditions:
    #     h5_file2 = '{sub}/{sub}_{cond}_posFilter_classical2.h5'.format(sub=sub, cond=cond) 
    #     data_tmp2 =  pd.read_hdf(h5_file2, 'data')
    #     data_tmp2.reset_index(inplace=True)
        
    #     if cond =='p50': # data files from p50 had wrong names
    #         new_val = {'HS': "LS", "LS": "HS"}
    #         data_tmp2.replace({"trial_vel": new_val}, inplace=True)

    #     idxcond = data_tmp[(data_tmp['cond']==cond)].index
    #     idxvel  = data_tmp2[(data_tmp2['trial'].isin(data_tmp.loc[idxcond,'trial']))].index

    #     data_tmp.loc[idxcond,'latency_x']      = list(data_tmp2.loc[idxvel, 'latency'])
    #     data_tmp.loc[idxcond,'ramp_pursuit_x'] = list(data_tmp2.loc[idxvel, 'a_pur']*1000)

    allSubsData = pd.concat([allSubsData,data_tmp], ignore_index=True)

allSubsData = allSubsData[allSubsData['trial']>10]

allSubsData.reset_index(inplace=True)

# plot difference HS-LS
anticipData = allSubsData.copy()
anticipData = anticipData.loc[~anticipData.cond_num.isin([0.0, 1.0]),['n1_tgVel','sub','cond_num','aSPv']]
meanAntiData   = anticipData.groupby(['sub', 'cond_num']).mean()
meanAntiDataN1 = anticipData.groupby(['sub', 'cond_num', 'n1_tgVel']).mean()
meanAntiDataN1.reset_index(inplace=True)
print(meanAntiData)
print(meanAntiDataN1)
for idx,row in meanAntiData.iterrows():
    sub, cond = idx
    idxAnti = (meanAntiDataN1['sub']==sub)&(meanAntiDataN1['cond_num']==cond)
    meanAntiDataN1.loc[idxAnti,'diff_aSPv-mean'] = meanAntiDataN1.loc[idxAnti,'aSPv'] - row['aSPv']

f = plt.figure(figsize=(single_col,6*cm))
sns.lineplot(data = meanAntiDataN1, x = 'cond_num', y = 'diff_aSPv-mean', hue='n1_tgVel',
             units="sub", estimator=None,style="sub", palette={'HS':colorHS,'LS':colorLS}
             )
grandMean = meanAntiDataN1.groupby(['cond_num', 'n1_tgVel']).mean()
grandMean.reset_index(inplace=True)
sns.scatterplot(data=grandMean, x='cond_num', y='diff_aSPv-mean', hue='n1_tgVel',palette={'HS':colorHS,'LS':colorLS})
plt.xlabel('P(HS)')
plt.ylabel('Difference from mean aSPv')
plt.xticks(cond_num[1:-1])
plt.xlim([0,1])
plt.tight_layout()
plt.savefig('{}/exp1_diff_aSPv-mean.png'.format(output_folder))
plt.savefig('{}/exp1_diff_aSPv-mean.pdf'.format(output_folder))
# plt.savefig('{}/exp1_diff_HS-LS.pdf'.format(output_folder))


# read LME results from csv generated on R
lmm_dir = "{}/LMM".format(output_folder)
lme_raneff     = pd.read_csv('{}/exp1_lmm_randomEffects.csv'.format(lmm_dir))
lme_fixeffAnti = pd.read_csv('{}/exp1_lmm_fixedeffectsAnti.csv'.format(lmm_dir))
lme_fixeffVGP  = pd.read_csv('{}/exp1_lmm_fixedeffectsVGP.csv'.format(lmm_dir))

lme_fixeffAnti.at[0,'Unnamed: 0'] = 'Intercept'
lme_fixeffVGP.at[0,'Unnamed: 0']  = 'Intercept'
lme_fixeffAnti.set_index('Unnamed: 0', inplace=True)
lme_fixeffVGP.set_index('Unnamed: 0', inplace=True)

lme_fixeffAnti.fillna(0, inplace=True)
lme_fixeffVGP.fillna(0, inplace=True)
lme_raneff.fillna(0, inplace=True)

# print(lme_fixeffAnti)
# print(lme_fixeffVGP)
# print(lme_raneff)

anticipParams = [['aSPon','Anticipation Onset', [-200,-100], 'Horizontal aSPon (ms)'],
                 ['aSPv', 'Anticipatory Eye Velocity', [-2.5,9], 'Horizontal aSPv (°/s)'],
                ]
anticipData = allSubsData.groupby(['sub','cond_num']).mean()
anticipData.reset_index(inplace=True)
cmapSpacing = [0.05, 0.12, 0.25, 0.5, 0.75, 0.88, 0.95]

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
        raneff      = lme_raneff.loc[(lme_raneff['sub']==sub)&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        
        reg = cond_num*(prob + s_prob) + (intercept + s_intercept)
        ax1.plot(cond_num, reg, color='grey', alpha=0.5)
    
    plotBoxDispersion(data=anticipData, 
                        by=['cond_num'], 
                        between=p[0], ax=ax2, alpha=0,
                        cmapAlpha=1,
                        scatterSize=25,
                        jitter=.01,
                        xticks= cond_num,
                        boxWidth = .045,
                        showfliers=False,
                        showKde=False,
                        cmapName='icefire',
                        cmapValues=cmapSpacing)
    plt.ylim(p[2])
    ax1.set_xlabel('P(HS)')
    ax1.set_ylabel(p[3])
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])

    plt.tight_layout()
    plt.savefig('{}/exp1_group_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/exp1_group_{}.png'.format(output_folder, p[1]))

# visParams = [['SPlat','Latency', [75,125], 'Horizontal SPlat (ms)'],
#              ['SPacc', 'Pursuit Acceleration', [35,150], 'Horizontal SPacc (°/s\N{SUPERSCRIPT TWO})'],
#             #  ['SPss', 'Steady State', [4, 17], 'Horizontal SPss (°/s)'],
#             ]
# visData = allSubsData.groupby(['sub','cond_num','trial_velocity']).agg(np.nanmean)
# visData.reset_index(inplace=True)

# idxHS   = [True if 'H' in x else False for x in visData['trial_velocity']]
# idxLS = [not x for x in idxHS]

# idxHS = np.array(idxHS)
# idxLS = np.array(idxLS)

# for p in visParams:
#     print(p[1])

#     intercept  = lme_fixeffVGP.loc['Intercept',p[0]]
#     prob       = lme_fixeffVGP.loc['prob',p[0]]
#     tgvel      = lme_fixeffVGP.loc['trial_velocityLS',p[0]]
#     prob_tgvel = lme_fixeffVGP.loc['trial_velocityLS:prob',p[0]]
    
#     fig1 = plt.subplots(figsize=(single_col, 6*cm)) # width, height
#     plt.suptitle(p[1])
#     ax1 = plt.subplot(1,1,1) 
#     ax2 = ax1.twiny() 
#     for sub in subjects:
#         raneff      = lme_raneff.loc[(lme_raneff['sub']==sub)&(lme_raneff['var']==p[0])]
#         s_intercept = list(raneff['Intercept'])[0]
#         s_prob      = list(raneff['prob'])[0]
#         s_tgvel     = list(raneff['trial_velocity'])[0]
        
#         regHS = (prob+s_prob)*cond_num[1:] + (tgvel+s_tgvel)*0 + prob_tgvel*0*cond_num[1:] + (intercept+s_intercept)
#         regLS = (prob+s_prob)*cond_num[:-1] + (tgvel+s_tgvel)*1 + prob_tgvel*1*cond_num[:-1] + (intercept+s_intercept)
#         ax1.plot(cond_num[1:]+.03, regHS, color=colorHS, alpha=0.7)
#         ax1.plot(cond_num[:-1]-.03, regLS, color=colorLS, alpha=0.7)
#     plotBoxDispersion(data=visData[idxHS], 
#                         by=['cond_num'], 
#                         between=p[0], ax=ax2, alpha=0,
#                         cmapAlpha=1,
#                         scatterSize=25,
#                         jitter=.01,
#                         xticks= cond_num[1:]+.03,
#                         boxWidth = .045,
#                         showfliers=False,
#                         showKde=False,
#                         color=colorHS,
#                         cmapName=None)
#     plotBoxDispersion(data=visData[idxLS], 
#                         by=['cond_num'], 
#                         between=p[0], ax=ax2, alpha=0,  
#                         cmapAlpha=1,
#                         boxWidth=.045,
#                         xticks=cond_num[:-1]-.03,
#                         scatterSize=25,
#                         jitter=.01,
#                         showfliers=False,
#                         showKde=False,
#                         color=colorLS,
#                         cmapName=None)
    
#     # plt.ylim(p[2])
#     ax1.set_xlabel('P(HS)')
#     ax1.set_ylabel(p[3])
#     ax1.set_xlim(np.array([-10,110])/100)
#     ax2.set_xlim(np.array([-10,110])/100)
#     ax1.set_xticks(cond_num)
#     ax1.set_xticklabels(cond_num)
#     ax2.set_xticks([])
#     # ax.set_xlabels(conditions)
#     # plt.xticks(rotation = 20)
#     ax1.legend(['HS', 'LS'])
   
#     plt.tight_layout()
    
#     plt.savefig('{}/exp1_group_{}.pdf'.format(output_folder, p[1]))
#     plt.savefig('{}/exp1_group_{}.png'.format(output_folder, p[1]))

lmm_dir = "{}/LMM".format(output_folder)
lme_raneff     = pd.read_csv('{}/exp1_lmm_n1Eff_randomEffects.csv'.format(lmm_dir))
lme_fixeffAnti = pd.read_csv('{}/exp1_lmm_n1Eff_fixedeffectsAnti.csv'.format(lmm_dir))

lme_fixeffAnti.at[0,'Unnamed: 0'] = 'Intercept'
lme_fixeffAnti.set_index('Unnamed: 0', inplace=True)

lme_fixeffAnti.fillna(0, inplace=True)
lme_raneff.fillna(0, inplace=True)

# print(lme_fixeffAnti)
# print(lme_raneff)

anticipParams = [['aSPon','Anticipation Onset', [-200,-100], 'Horizontal aSPon (ms)'],
                 ['aSPv', 'Anticipatory Eye Velocity', [-2.5,9], 'Horizontal aSPv (°/s)'],
                ]
anticipData = allSubsData.groupby(['sub','cond_num','n1_tgVel']).mean()
anticipData.reset_index(inplace=True)

idxHS   = [True if 'H' in x else False for x in anticipData['n1_tgVel']]
idxLS = [not x for x in idxHS]

idxHS = np.array(idxHS)
idxLS = np.array(idxLS)


for p in anticipParams:
    print(p[1])

    intercept = lme_fixeffAnti.loc['Intercept',p[0]]
    prob      = lme_fixeffAnti.loc['prob',p[0]]
    n1_tgVel  = lme_fixeffAnti.loc['n1_velLS',p[0]]
    prob_n1   = lme_fixeffAnti.loc['n1_velLS:prob',p[0]]
    
    # fig1 = plt.figure(figsize=(single_col*0.5, 3*cm)) 
    f,ax1 = plt.subplots(figsize=(single_col, 6*cm))# width, height
    plt.suptitle(p[1])
    ax1 = plt.subplot(1,1,1)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==sub)&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_prob      = list(raneff['prob'])[0]
        s_n1Vel     = list(raneff['n1_tgVel'])[0]
        
        reg_n1HS = cond_num[1:]*(prob + s_prob) + (intercept + s_intercept)
        reg_n1LS = cond_num[:-1]*(prob_n1 + prob + s_prob) + (s_n1Vel + n1_tgVel) + (intercept + s_intercept)
        ax1.plot(cond_num[1:]+.03, reg_n1HS, color=colorHS, alpha=0.5)
        ax1.plot(cond_num[:-1]-.03, reg_n1LS, color=colorLS, alpha=0.5)
    
    plotBoxDispersion(data=anticipData[idxHS], 
                        by=['cond_num'], 
                        between=p[0], ax=ax2, alpha=0,
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
                        by=['cond_num'], 
                        between=p[0], ax=ax2, alpha=0,  
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
    ax1.set_xlabel('P(HS)')
    ax1.set_ylabel(p[3])
    ax1.set_xlim(np.array([-10,110])/100)
    ax2.set_xlim(np.array([-10,110])/100)
    ax1.set_xticks(cond_num)
    ax1.set_xticklabels(cond_num)
    ax2.set_xticks([])
    ax1.legend(['N-1=HS', 'N-1=LS'])

    plt.tight_layout()
    plt.savefig('{}/exp1_group_n1Eff_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/exp1_group_n1Eff_{}.png'.format(output_folder, p[1]))
