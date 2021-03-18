# -*- coding: utf-8 -*-
"""
@author: leonlwang@tencent.com
"""

import os,sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch
import seaborn as sns # for boxplot
import matplotlib.gridspec as gridspec # for tight layout
import pdb

import logging
import time

from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"  #'sans-serif'


def draw_bbox(ax, bb):
    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                            abs(bb.width), abs(bb.height),
                            boxstyle="square,pad=0.",
                            ec="k", fc="none", zorder=10.,
                            )
    ax.add_patch(p_bbox)

def IC_class_to_group_1D(data_list,cutoff):
    data_list=np.asarray(data_list)
    ic_reslut_N_class=np.zeros_like(data_list)
    ic_reslut_N_class.astype(np.uint8)
    for i in range(0,data_list.shape[0]):
        for kk in range(0,len(cutoff)-1):
            if data_list[i]>=cutoff[kk] and data_list[i]<cutoff[kk+1]:
                ic_reslut_N_class[i]=int(kk)
        if data_list[i]>=cutoff[-1]:
            ic_reslut_N_class[i]=int(len(cutoff)-1)
    return ic_reslut_N_class

def get_num_of_same_ver1(ic_dict_ai,img_list_in_order,user31_result,DIFF_SCORE_THRESHOLD, DATA_CLASS):
    nbr_same_scoring_case=np.zeros([31])
    for user_index in range(0,31):
        one_user_result=dict()
        for one_key, one_value in user31_result.items():
            one_user_result.update({one_key: one_value[user_index]})
    
        #check user with AI
        ic_one_user=list()
        ic_ai=list()
        for i in range(0,len(img_list_in_order)):
            img_name=img_list_in_order[i]
            ic_one_user.append(one_user_result[img_name])
            ic_ai.append(ic_dict_ai[img_name])
        
        if DATA_CLASS=='CONTINUE':
            ic_one_user = np.round(np.array(ic_one_user),decimals=2)
            ic_ai = np.round(np.array(ic_ai),decimals=2)
        elif DATA_CLASS=='4CLASS':
            cutoff_4class=[0,1,5,10]
            ic_one_user = IC_class_to_group_1D(ic_one_user,cutoff=cutoff_4class)
            ic_ai       = IC_class_to_group_1D(ic_ai,cutoff=cutoff_4class)
        elif DATA_CLASS=='2CLASS':
            cutoff_2class=[0,1]
            ic_one_user = IC_class_to_group_1D(ic_one_user,cutoff=cutoff_2class)
            ic_ai       = IC_class_to_group_1D(ic_ai,cutoff=cutoff_2class)
        
        ic_diff=abs(ic_one_user-ic_ai)
        nbr_of_diff = len(np.where(ic_diff>DIFF_SCORE_THRESHOLD)[0])
        nbr_same_scoring_case[user_index]= 109 - nbr_of_diff
    return nbr_same_scoring_case

def get_num_of_same(ic_dict_ai,img_list_in_order,user31_result,DIFF_SCORE_THRESHOLD, DATA_CLASS):
    nbr_same_scoring_case=np.zeros([31])
    for user_index in range(0,31):
        print("_"*100)
        print('user_index = ',user_index)
    
        one_user_result=dict()
        for one_key, one_value in user31_result.items():
            one_user_result.update({one_key: one_value[user_index]})
    
        #check user with AI
        ic_one_user=list()
        ic_ai=list()
        for i in range(0,len(img_list_in_order)):
            img_name=img_list_in_order[i]
            ic_one_user.append(one_user_result[img_name])
            ic_ai.append(ic_dict_ai[img_name])
        
        if DATA_CLASS=='CONTINUE':
            ic_one_user = np.round(np.array(ic_one_user),decimals=6)
            ic_ai = np.round(np.array(ic_ai),decimals=6)
            ic_diff_value=abs(ic_one_user-ic_ai)
            ic_diff = np.zeros_like(ic_diff_value)
            ic_diff[ic_diff_value>DIFF_SCORE_THRESHOLD] = 1
            
            cutoff_2class=[0,1]
            ic_one_user_2c = IC_class_to_group_1D(ic_one_user,cutoff=cutoff_2class)
            ic_ai_2c       = IC_class_to_group_1D(ic_ai,cutoff=cutoff_2class)
            ic_diff_2c = (np.logical_xor(ic_one_user_2c,ic_ai_2c)).astype(np.uint8)

            cutoff_4class=[0,1,5,10]
            ic_one_user_4c = IC_class_to_group_1D(ic_one_user,cutoff=cutoff_4class)
            ic_ai_4c       = IC_class_to_group_1D(ic_ai,cutoff=cutoff_4class)
            ic_diff_4c = (np.logical_xor(ic_one_user_4c,ic_ai_4c)).astype(np.uint8)
            
            temp_sum = np.logical_xor(ic_diff_4c,ic_diff_2c).astype(np.uint8)
            print('if 4C==2C ? :',temp_sum.sum())
            if temp_sum.sum()>0:
                print('temp_sum.sum()>0')
                print(tttt)

            result = ((ic_diff+ic_diff_4c+ic_diff_2c).astype(np.bool)).astype(np.uint8)
            nbr_of_diff = result.sum()

            
        elif DATA_CLASS=='4CLASS':
            cutoff_4class=[0,1,5,10]
            ic_one_user = IC_class_to_group_1D(ic_one_user,cutoff=cutoff_4class)
            ic_ai       = IC_class_to_group_1D(ic_ai,cutoff=cutoff_4class)
            
            ic_diff=abs(ic_one_user-ic_ai)
            nbr_of_diff = len(np.where(ic_diff>DIFF_SCORE_THRESHOLD)[0])
        
        elif DATA_CLASS=='2CLASS':
            cutoff_2class=[0,1]
            ic_one_user = IC_class_to_group_1D(ic_one_user,cutoff=cutoff_2class)
            ic_ai       = IC_class_to_group_1D(ic_ai,cutoff=cutoff_2class)

            ic_diff=abs(ic_one_user-ic_ai)
            nbr_of_diff = len(np.where(ic_diff>DIFF_SCORE_THRESHOLD)[0])
        
        nbr_same_scoring_case[user_index]= 109 - nbr_of_diff
    return nbr_same_scoring_case

def show_HMP_percent(nbr_same_scoring_case):
    repeat_ai_score=np.zeros([4])
    repeat_ai_score[0] = nbr_same_scoring_case.mean()/109
    repeat_ai_score[1] = nbr_same_scoring_case[0:11].mean()/109
    repeat_ai_score[2] = nbr_same_scoring_case[11:21].mean()/109
    repeat_ai_score[3] = nbr_same_scoring_case[21:31].mean()/109
    
    # repeat_ai_score=np.ceil(100.0*np.round(repeat_ai_score,decimals=4))
    repeat_ai_score=100.0*np.round(repeat_ai_score,decimals=4)
    print('def show_HMP_percent(): repeat_ai_score = ',repeat_ai_score)
    return repeat_ai_score
    # return str(repeat_ai_score[0])+'%,'+str(repeat_ai_score[1])+'%,'+str(repeat_ai_score[2])+'%,'+str(repeat_ai_score[3])+'%'

def make_fig_acceptance(nbr_same_scoring_case, TITLE_NAME):
    nbr_same_scoring_case = 100*nbr_same_scoring_case/109.0
    plt.close('all')
    title=['All', 'Senior','Intermediate','Junior']
    fig=plt.figure()
    fig.set_size_inches(7, 3)
    FONTSIZE=14
    fig.suptitle(TITLE_NAME,fontsize=FONTSIZE)
    continent_colors=["w","w","w"]
    
    YLIM_RANGE=nbr_same_scoring_case.max()-nbr_same_scoring_case.min()
    YLIM=[nbr_same_scoring_case.min()-0.05*YLIM_RANGE,nbr_same_scoring_case.max()+0.05*YLIM_RANGE]

    for i in range(0,4):
        plt.subplot(1,4,i+1)
        plt.subplots_adjust(wspace = .00001)
        if i==0:
            fig_array=nbr_same_scoring_case.transpose()
            fig_df= pd.DataFrame(fig_array,columns=[title[i]])
    

            bplot=sns.boxplot(data=fig_df,width=0.8,
                              showmeans=True,
                              meanprops={"marker":"o",
                              "markerfacecolor":"white", 
                              "markeredgecolor":"black",
                              "markersize":"8"})
            for j in range(0,1):
                mybox = bplot.artists[j]
                mybox.set_facecolor(continent_colors[j])
            bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
            plt.tick_params(labelsize=FONTSIZE)
            ax=plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.ylabel('Percentage of accepted\nAI scores (%)',fontsize=FONTSIZE)
            plt.ylim(YLIM)
        elif i==1:
            fig_array= nbr_same_scoring_case[0:11].transpose()
            fig_df= pd.DataFrame(fig_array,columns=[title[i]])
            bplot=sns.boxplot(data=fig_df,width=0.8,
                              showmeans=True,
                              meanprops={"marker":"o",
                              "markerfacecolor":"white", 
                              "markeredgecolor":"black",
                              "markersize":"8"})
            for j in range(0,1):
                mybox = bplot.artists[j]
                mybox.set_facecolor(continent_colors[j])
            bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
            plt.tick_params(labelsize=FONTSIZE)
            ax=plt.gca()
            # ax.set_title(title[i]+' '+'pathologists',fontsize=FONTSIZE)
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.ylim(YLIM)
        elif i==2:
            fig_array= nbr_same_scoring_case[11:21].transpose()
            fig_df= pd.DataFrame(fig_array,columns=[title[i]])
            bplot=sns.boxplot(data=fig_df,width=0.8,
                              showmeans=True,
                              meanprops={"marker":"o",
                              "markerfacecolor":"white", 
                              "markeredgecolor":"black",
                              "markersize":"8"})
            for j in range(0,1):
                mybox = bplot.artists[j]
                mybox.set_facecolor(continent_colors[j])
            bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
            plt.tick_params(labelsize=FONTSIZE)
            ax=plt.gca()
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.ylim(YLIM)
        elif i==3:
            fig_array=nbr_same_scoring_case[21:31].transpose()
            fig_df= pd.DataFrame(fig_array,columns=[title[i]])
            bplot=sns.boxplot(data=fig_df,width=0.8,
                              showmeans=True,
                              meanprops={"marker":"o",
                              "markerfacecolor":"white", 
                              "markeredgecolor":"black",
                              "markersize":"8"})
            for j in range(0,1):
                mybox = bplot.artists[j]
                mybox.set_facecolor(continent_colors[j])
            bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
            plt.tick_params(labelsize=FONTSIZE)
            ax=plt.gca()
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['top'].set_color('none')
            plt.ylim(YLIM)
    return fig


###############################################################################
###############################################################################
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y_%b_%d_%Hh%M")
print('Current Timestamp : ', timestampStr)
logging.info('\n\nStart of run result:')
result_Calc_dir = './result_Calc'
result_Fig_dir = './result_Fig'
cutoff_4class=[0,1,5,10]
cutoff_2class=[0,1]
EXP=3
###############################################################################
## Continue Score
###############################################################################
## [1] load AI
ic_dict_ai = json.load(open("./Result_AI/IC_SCORE_AI_ImgID.csv.json",'r'))
img_list_in_order=np.load('./IC_Score_Pathologist/img_list_in_order.npy').tolist()

## [2] load algo, with pre-process 4 clsss, 2 class
user31_result = json.load(open("./IC_Score_Pathologist/ic_dict_medicine_exp"+str(EXP)+".json",'r'))

DATA_CLASS=['CONTINUE','4CLASS','2CLASS']
nbr_same_scoring_case_th1 = get_num_of_same(ic_dict_ai,img_list_in_order,user31_result,DIFF_SCORE_THRESHOLD=1, DATA_CLASS='CONTINUE')
nbr_same_scoring_case_th5 = get_num_of_same(ic_dict_ai,img_list_in_order,user31_result,DIFF_SCORE_THRESHOLD=5, DATA_CLASS='CONTINUE')
nbr_same_scoring_case_2C = get_num_of_same(ic_dict_ai,img_list_in_order,user31_result,DIFF_SCORE_THRESHOLD=0, DATA_CLASS='2CLASS')
nbr_same_scoring_case_4C = get_num_of_same(ic_dict_ai,img_list_in_order,user31_result,DIFF_SCORE_THRESHOLD=0, DATA_CLASS='4CLASS')

np.savetxt(os.path.join(result_Calc_dir,'nbr_same_scoring_case_th1.out'), nbr_same_scoring_case_th1,fmt='%d')
np.savetxt(os.path.join(result_Calc_dir,'nbr_same_scoring_case_th5.out'), nbr_same_scoring_case_th5,fmt='%d')
np.savetxt(os.path.join(result_Calc_dir,'nbr_same_scoring_case_2C.out'), nbr_same_scoring_case_2C,fmt='%d')
np.savetxt(os.path.join(result_Calc_dir,'nbr_same_scoring_case_4C.out'), nbr_same_scoring_case_4C,fmt='%d')

accept_AI_score_percentage_ROW_th1_th5_2C_4C_COL_all_H_M_P = np.array([show_HMP_percent(nbr_same_scoring_case_th1),
                                                                       show_HMP_percent(nbr_same_scoring_case_th5),
                                                                       show_HMP_percent(nbr_same_scoring_case_2C),
                                                                       show_HMP_percent(nbr_same_scoring_case_4C)
                                                                       ])
np.savetxt(os.path.join(result_Calc_dir,'accept_AI_score_percentage_ROW_th1_th5_2C_4C_COL_all_H_M_P.out'), accept_AI_score_percentage_ROW_th1_th5_2C_4C_COL_all_H_M_P,fmt='%.4f')

print('nbr_same_scoring_case_th1 = ',nbr_same_scoring_case_th1)
print('nbr_same_scoring_case_th5 = ',nbr_same_scoring_case_th5)
print('nbr_same_scoring_case_2C =',nbr_same_scoring_case_2C)
print('nbr_same_scoring_case_4C =',nbr_same_scoring_case_4C)
print('nbr_same_scoring_case_th1.mean() = '+str(nbr_same_scoring_case_th1.mean()))
print('nbr_same_scoring_case_th5.mean() = '+str(nbr_same_scoring_case_th5.mean()))
print('nbr_same_scoring_case_2C.mean() ='+str(nbr_same_scoring_case_2C.mean()))
print('nbr_same_scoring_case_4C.mean() ='+str(nbr_same_scoring_case_4C.mean()))
print('accept_AI_score_percentage_ROW_th1_th5_2C_4C_COL_all_H_M_P = \n')
print(accept_AI_score_percentage_ROW_th1_th5_2C_4C_COL_all_H_M_P)


##############################################################################
## Calculate for figures
###############################################################################
plt.close('all')
DIFF_SCORE_THRESHOLD=1
TYPE='CONTINUE'
Fig_Title='Fully acceptance'
fig = make_fig_acceptance(nbr_same_scoring_case_th1,Fig_Title)
fig.savefig(os.path.join(result_Fig_dir ,'Fig6a_Acceptance_of_AI_boxplot_DIFF_SCORE_THRESHOLD_'+str(DIFF_SCORE_THRESHOLD)+'_'+TYPE+'.pdf'),dpi=300)
fig.savefig(os.path.join(result_Fig_dir ,'Fig6a_Acceptance_of_AI_boxplot_DIFF_SCORE_THRESHOLD_'+str(DIFF_SCORE_THRESHOLD)+'_'+TYPE+'.png'),dpi=300)

DIFF_SCORE_THRESHOLD=5
TYPE='CONTINUE'
Fig_Title='Almost acceptance'
fig = make_fig_acceptance(nbr_same_scoring_case_th5,Fig_Title)
fig.savefig(os.path.join(result_Fig_dir ,'Fig6b_Acceptance_of_AI_boxplot_DIFF_SCORE_THRESHOLD_'+str(DIFF_SCORE_THRESHOLD)+'_'+TYPE+'.pdf'),dpi=300)
fig.savefig(os.path.join(result_Fig_dir ,'Fig6b_Acceptance_of_AI_boxplot_DIFF_SCORE_THRESHOLD_'+str(DIFF_SCORE_THRESHOLD)+'_'+TYPE+'.png'),dpi=300)

DIFF_SCORE_THRESHOLD=0
TYPE='2CLASS'
Fig_Title='2-category acceptance'
fig = make_fig_acceptance(nbr_same_scoring_case_2C,Fig_Title)
fig.savefig(os.path.join(result_Fig_dir ,'Fig6c_Acceptance_of_AI_boxplot_DIFF_SCORE_THRESHOLD_'+str(DIFF_SCORE_THRESHOLD)+'_'+TYPE+'.pdf'),dpi=300)
fig.savefig(os.path.join(result_Fig_dir ,'Fig6c_Acceptance_of_AI_boxplot_DIFF_SCORE_THRESHOLD_'+str(DIFF_SCORE_THRESHOLD)+'_'+TYPE+'.png'),dpi=300)

DIFF_SCORE_THRESHOLD=0
TYPE='4CLASS'
Fig_Title='4-category acceptance'
fig = make_fig_acceptance(nbr_same_scoring_case_4C,Fig_Title)
fig.savefig(os.path.join(result_Fig_dir ,'Fig6d_Acceptance_of_AI_boxplot_DIFF_SCORE_THRESHOLD_'+str(DIFF_SCORE_THRESHOLD)+'_'+TYPE+'.pdf'),dpi=300)
fig.savefig(os.path.join(result_Fig_dir ,'Fig6d_Acceptance_of_AI_boxplot_DIFF_SCORE_THRESHOLD_'+str(DIFF_SCORE_THRESHOLD)+'_'+TYPE+'.png'),dpi=300)

