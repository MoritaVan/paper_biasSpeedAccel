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

main_dir = "../data/"
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
                'V1-100_V0-0', 
                'V2-100_V0-0',
                'V3-100_V0-0',
                ]

cond100 =   {
                'V1-100_V0-0': 'v11', 
                'V2-100_V0-0': 'v22',
                'V3-100_V0-0': 'v33',
                'Va-100_V0-0': 'vacc', 
                'Vd-100_V0-0': 'vdec',
}
v0 =   {
                'V1-100_V0-0': np.round(11/np.sqrt(2),2), 
                'V2-100_V0-0': np.round(22/np.sqrt(2),2),
                'V3-100_V0-0': np.round(33/np.sqrt(2),2),
                'Va-100_V0-0': np.round(11/np.sqrt(2),2), 
                'Vd-100_V0-0': np.round(33/np.sqrt(2),2),
}
ac =   {
                'V1-100_V0-0': 0, 
                'V2-100_V0-0': 0,
                'V3-100_V0-0': 0,
                'Va-100_V0-0': np.round(22/np.sqrt(2),2), 
                'Vd-100_V0-0': np.round(-22/np.sqrt(2),2),
}


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
    data_tmp['cond'] = [cond100[x] for x in data_tmp['condition']]
    data_tmp['v0']   = [v0[x] for x in data_tmp['condition']]
    data_tmp['ac']   = [ac[x] for x in data_tmp['condition']]
    
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
        data_tmp['cond'] = [cond100[x] for x in data_tmp['condition']]
        data_tmp['v0']   = [v0[x] for x in data_tmp['condition']]
        data_tmp['ac']   = [ac[x] for x in data_tmp['condition']]
        
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
allSubsData.reset_index(inplace=True)

# read LME results from csv generated on R
lme_raneff     = pd.read_csv('{}/exp2_cond100Pred_lmm_randomEffects.csv'.format(lmm_dir_exp))
lme_fixeffAnti = pd.read_csv('{}/exp2_cond100Pred_lmm_fixedeffectsAnti.csv'.format(lmm_dir_exp))
lme_fixeffVGP  = pd.read_csv('{}/exp2_cond100Pred_lmm_fixedeffectsVGP.csv'.format(lmm_dir_exp))

lme_fixeffAnti.at[0,'Unnamed: 0'] = 'Intercept'
lme_fixeffVGP.at[0,'Unnamed: 0']  = 'Intercept'
lme_fixeffAnti.set_index('Unnamed: 0', inplace=True)
lme_fixeffVGP.set_index('Unnamed: 0', inplace=True)

lme_fixeffAnti.fillna(0, inplace=True)
lme_fixeffVGP.fillna(0, inplace=True)
lme_raneff.fillna(0, inplace=True)


v0_val = np.array([7.78, 15.56, 23.33])
ac_val = np.array([15.56, 0, -15.56])
ac_plot = [v0_val[0]+1.5, v0_val[-1]-1.5]

anticipParams = [['aSPon','Anticipation Onset', [-150,0], ' aSPon (ms)'],
                 ['aSPv', 'Anticipatory Eye Velocity', [-2.5,10], ' aSPv (°/s)'],
                ]
anticipData = allSubsData.groupby(['exp', 'sub','v0','ac']).mean()
anticipData.reset_index(inplace=True)
print(lme_fixeffAnti)
print(lme_fixeffVGP)
print(lme_raneff)

idxConst = [True if x==0.0 else False for x in anticipData['ac']]
idxAccel = [not x for x in idxConst]

idxAccel = np.array(idxAccel)
idxConst = np.array(idxConst)

idxExp = [True if 'Duration' in x else False for x in anticipData['exp']]
idxCtrl = [not x for x in idxExp]

for p in anticipParams:
    print(p[1])

    intercept = lme_fixeffAnti.loc['Intercept',p[0]]
    v0        = lme_fixeffAnti.loc['v0',p[0]]
    ac        = lme_fixeffAnti.loc['accel',p[0]]
    v0_ac     = lme_fixeffAnti.loc['v0:accel',p[0]]
    
    f,ax1 = plt.subplots(figsize=(two_col, 8*cm))# width, height
    plt.suptitle(p[1])

    # EXP const displacement
    var2plot = '{}_x'.format(p[0])
    y_label  = '{}'.format(p[3])
    ax1 = plt.subplot(1,2,1)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    plt.title('Model: Initial speed')
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-2:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_v0        = list(raneff['v0'])[0]
        s_ac        = list(raneff['accel'])[0]
        
        reg_Const = v0_val*(v0 + s_v0) + (intercept + s_intercept)
        reg_Accel = v0_val*(v0 + s_v0) + ac_val*(s_ac + ac) + v0_val*ac_val*(v0_ac) + (intercept + s_intercept)
        ax1.plot(v0_val, reg_Const, color=colorLS, alpha=0.5)
    
    plotBoxDispersion(data=anticipData[idxConst], 
                        by=['v0'], 
                        between=var2plot, ax=ax2, alpha=0,
                        cmapAlpha=1,
                        scatterSize=25,
                        jitter=0.25,
                        xticks= v0_val,
                        boxWidth = 1,
                        showfliers=False,
                        showKde=False,
                        color = colorLS,
                        cmapName=None)
    plotBoxDispersion(data=anticipData[idxAccel], 
                        by=['ac'], 
                        between=var2plot, ax=ax2, alpha=0,  
                        cmapAlpha=1,
                        boxWidth=1,
                        xticks=ac_plot,
                        groups=[15.56, -15.56],
                        scatterSize=25,
                        jitter=0.25,
                        showfliers=False,
                        showKde=False,
                        color = colorHS,
                        cmapName=None)
    plt.ylim(p[2])
    ax1.set_xlabel('Initial Speed')
    ax1.set_ylabel(y_label)
    ax1.set_xlim([v0_val[0]-2,v0_val[-1]+2])
    ax2.set_xlim([v0_val[0]-2,v0_val[-1]+2])
    ax1.set_xticks(v0_val)
    ax1.set_xticklabels(['v11', 'v22', 'v33'])
    ax2.set_xticks([])

    ax1 = plt.subplot(1,2,2)
    ax2 = ax1.twiny() # applies twinx to ax2, which is the second y axis.
    plt.title('Model: Initial speed + acceleration')
    for sub in subjects:
        raneff      = lme_raneff.loc[(lme_raneff['sub']==int(sub[-2:]))&(lme_raneff['var']==p[0])]
        s_intercept = list(raneff['Intercept'])[0]
        s_v0        = list(raneff['v0'])[0]
        s_ac        = list(raneff['accel'])[0]
        
        reg_Const = v0_val*(v0 + s_v0) + (intercept + s_intercept)
        reg_Accel = v0_val*(v0 + s_v0) + ac_val*(s_ac + ac) + v0_val*ac_val*(v0_ac) + (intercept + s_intercept)
        ax1.plot(v0_val, reg_Accel, color=colorHS, alpha=0.5)
    
    plotBoxDispersion(data=anticipData[idxConst], 
                        by=['v0'], 
                        between=var2plot, ax=ax2, alpha=0,
                        cmapAlpha=1,
                        scatterSize=25,
                        jitter=0.25,
                        xticks= v0_val,
                        boxWidth = 1,
                        showfliers=False,
                        showKde=False,
                        color = colorLS,
                        cmapName=None)
    plotBoxDispersion(data=anticipData[idxAccel], 
                        by=['ac'], 
                        between=var2plot, ax=ax2, alpha=0,  
                        cmapAlpha=1,
                        boxWidth=1,
                        xticks=ac_plot,
                        groups=[15.56, -15.56],
                        scatterSize=25,
                        jitter=0.25,
                        showfliers=False,
                        showKde=False,
                        color = colorHS,
                        cmapName=None)
    plt.ylim(p[2])
    ax1.set_xlabel('Initial Speed')
    ax1.set_ylabel(y_label)
    ax1.set_xlim([v0_val[0]-2,v0_val[-1]+2])
    ax2.set_xlim([v0_val[0]-2,v0_val[-1]+2])
    ax1.set_xticks(v0_val)
    ax1.set_xticklabels(['v11', 'v22', 'v33'])
    ax2.set_xticks([])
    # ax1.legend(['','','Const.', 'Accel.'])

    plt.tight_layout()
    plt.savefig('{}/exp2_cond100Pred_group_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/exp2_cond100Pred_group_{}.png'.format(output_folder, p[1]))


cmap_blues = plt.get_cmap('Blues_r') 
cmap_reds = plt.get_cmap('Reds_r') 

blues2 = cmap_blues(np.linspace(0.2, .6, 3))
reds2  = cmap_reds(np.linspace(0.2, .6, 2))
colors100 = {
    'v11': blues2[0],
    'v22': blues2[1],
    'v33': blues2[2],
    'vacc': reds2[0],
    'vdec': reds2[1],
}

pairs = [
            ['v11', 'v22'], 
            ['v11', 'v33'],   
            ['v22', 'v33'],
            ['v11', 'vacc'], 
            ['v22', 'vacc'],   
            ['v33', 'vacc'],
            ['v11', 'vdec'], 
            ['v22', 'vdec'],   
            ['v33', 'vdec'],
            ['vacc', 'vdec']
         ]
pvalues_antivel = [
    .00001, # 1-2
    .00001, # 1-3
    .0822, # 2-3
    .3791, # a-1
    .0196, # a-2
    .00001, # a-3 
    .00001, # d-1
    .1634, # d-2
    .1753, # d-3
    .0005, # a-d
]

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "***"
    elif pvalue <= 0.001:
        return "**"
    elif pvalue <= 0.01:
        return "*"
    return "ns"

pvalues_antivel   = [convert_pvalue_to_asterisks(x) for x in pvalues_antivel]

anticipParams = [
                 ['aSPv', 'Anticipatory Eye Velocity', [-2.5,10], ' aSPv (°/s)'],
                ]
anticipData = allSubsData.groupby(['sub','cond']).mean()
anticipData.reset_index(inplace=True)
print(anticipData.head(10))

for p in anticipParams:
    print(p[1])

    formatted_pvalues = pvalues_antivel

    fig1 = plt.figure(figsize=(two_col, 7*cm)) # width, height
    plt.suptitle(p[1])
    ax1 = plt.subplot(1,2,1)
    var2plot = '{}_x'.format(p[0])
    label = 'Horizontal {}'.format(p[3])
    plotBoxDispersion(data=anticipData, 
                      by='cond', 
                    between=var2plot, ax=ax1, alpha=0,
                    groups=list(cond100.values()),
                    groupsNames=list(cond100.values()),
                    scatterSize=30,
                    cmapAlpha=1,
                    jitter=0.1,
                    boxWidth = 0.4,
                    showfliers=False,
                    showKde=False,
                    cmap=list(colors100.values())
                    )

    plt.ylim(p[2])
    ax1.set_xlabel('Condition')
    ax1.set_ylabel(label)
    ax1.patch.set_visible(False)

    ymin, ymax = ax1.get_ylim()
    pos      = ax1.get_xticks()
    labels   = ax1.get_xticklabels()
    labels   = [l.get_text() for l in labels]
    groupPos = {x:y for x,y in zip(labels,pos)}
    pairPos  = [[groupPos[x[0]],groupPos[x[1]]] for x in pairs]
    prop = (ymax-ymin)/20
    h0 = ymax-(prop/2)
    for idx,pval in enumerate(formatted_pvalues):
        if pval != 'ns':
            xmin = pairPos[idx][0]
            xmax = pairPos[idx][1]
            xtxt = (xmin+xmax)/2 - .05
            ytxt = h0-.1
            plt.hlines(y=h0, xmin=xmin, xmax=xmax, colors='k', linewidth=.5)
            ax1.annotate(pval, xy=(xtxt,ytxt), zorder=10, fontsize=6)
            h0 += prop

    plt.ylim([ymin,h0+.01])
    
    ax1 = plt.subplot(1,2,2) 
    var2plot = '{}_y'.format(p[0])
    label = 'Vertical {}'.format(p[3])
    plotBoxDispersion(data=anticipData, 
                      by='cond', 
                    between=var2plot, ax=ax1, alpha=0, 
                    groups=list(cond100.values()),
                    groupsNames=list(cond100.values()),
                    scatterSize=30,
                    cmapAlpha=1,
                    jitter=0.1,
                    boxWidth = 0.4,
                    showfliers=False,
                    showKde=False,
                    cmap=list(colors100.values()))
    plt.ylim(p[2])
    ax1.set_xlabel('Condition')
    ax1.set_ylabel(label)
    ax1.patch.set_visible(False)    

    ymin, ymax = ax1.get_ylim()
    pos      = ax1.get_xticks()
    labels   = ax1.get_xticklabels()
    labels   = [l.get_text() for l in labels]
    groupPos = {x:y for x,y in zip(labels,pos)}
    pairPos  = [[groupPos[x[0]],groupPos[x[1]]] for x in pairs]
    prop = (ymax-ymin)/20
    h0 = ymax-(prop/2)
    for idx,pval in enumerate(formatted_pvalues):
        if pval != 'ns':
            xmin = pairPos[idx][0]
            xmax = pairPos[idx][1]
            xtxt = (xmin+xmax)/2 - .05
            ytxt = h0-.1
            plt.hlines(y=h0, xmin=xmin, xmax=xmax, colors='k', linewidth=.5)
            ax1.annotate(pval, xy=(xtxt,ytxt), zorder=10, fontsize=6)
            h0 += prop

    plt.ylim([ymin,h0+.01])

    plt.tight_layout()
    
    plt.savefig('{}/exp2_cond100Pred_groupPairwise_{}.pdf'.format(output_folder, p[1]))
    plt.savefig('{}/exp2_cond100Pred_groupPairwise_{}.png'.format(output_folder, p[1]))
