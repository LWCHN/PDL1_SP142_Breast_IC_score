# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:44:47 2020

@author: leonlwang
"""
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y_%b_%d_%Hh%M")
print('Current Timestamp : ', timestampStr)
import seaborn as sns

from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"  #'sans-serif'


def make_boxplot(fig_array,PDF_NAME,xlabelname,ylabelname,TITLE,YLIM):
    FONTSIZE=12
    plt.close("all")
    fig_df= pd.DataFrame(fig_array,columns=[' RS1','RS2','RS3 '])
    fig99, ax = plt.subplots()
    fig99.set_size_inches(4, 3.5)
    continent_colors=["r","g","b"]
    bplot=sns.boxplot(data=fig_df, 
                     width=0.5)
    for i in range(0,3):
        mybox = bplot.artists[i]
        mybox.set_facecolor(continent_colors[i])
    bplot = sns.stripplot(data=fig_df,
                          jitter=True, marker='o',
                          alpha=0.5, 
                          color="black")
    plt.ylabel(ylabelname,fontsize=FONTSIZE)
    plt.xlabel(xlabelname,fontsize=FONTSIZE)
    plt.title(TITLE,fontsize=FONTSIZE)
    plt.tick_params(labelsize=FONTSIZE)
    plt.grid(linestyle="--", alpha=0.2)
    ax.grid(True, which='both')
    plt.ylim(YLIM)
    fig99.tight_layout()
    fig99.savefig(PDF_NAME,dpi=400)
    
    
result_Calc_dir='./result_Calc'
save_dir='./result_Fig'

params = [(2, [0.65, 1.0], 'Boxplot accuracy for 2 categories'),
           (4, [0.35, 0.95], 'Boxplot accuracy for 4 categories')
          ]
for param in params:
    CLASS, YLIM, SUPTITLE = param
    print("CLASS, YLIM, SUPTITLE = ",CLASS, YLIM, SUPTITLE)


for param in params:
    CLASS, YLIM, SUPTITLE = param
    
    df_e1_c4 = pd.read_excel(os.path.join(result_Calc_dir,'evaluate_user31_exp1_'+str(CLASS)+'class.csv.xlsx'))
    array_e1_c4=df_e1_c4.to_numpy()
    df_e2_c4 = pd.read_excel(os.path.join(result_Calc_dir,'evaluate_user31_exp2_'+str(CLASS)+'class.csv.xlsx'))
    array_e2_c4=df_e1_c4.to_numpy()
    df_e3_c4 = pd.read_excel(os.path.join(result_Calc_dir,'evaluate_user31_exp3_'+str(CLASS)+'class.csv.xlsx'))
    array_e3_c4=df_e1_c4.to_numpy()
    
    plt.close("all")
    ## fig ACC
    DATA_NAME='acc'
    Y_LABEL_NAME = 'Accuracy'
    
    data_all =np.zeros([31,3])
    data_all[:,0]=df_e1_c4[DATA_NAME].to_numpy()
    data_all[:,1]=df_e2_c4[DATA_NAME].to_numpy()
    data_all[:,2]=df_e3_c4[DATA_NAME].to_numpy()
    data_H=data_all[0:11,:]
    data_M=data_all[11:21,:]
    data_P=data_all[21:31,:]
    
    
    ####################################################
    plt.close('all')
    ylabelname='Accuracy'
    xlabelname=''
    title=['All', 'Senior','Intermediate','Junior']
    fig=plt.figure(num='Acc for all and 3 group H,M,P')
    fig.set_size_inches(7.5, 4.5)
    FONTSIZE=14
    fig.suptitle(SUPTITLE,fontsize=FONTSIZE)
    fig.subplots_adjust(top=0.85)
    continent_colors = ['pink', 'lightgreen','lightblue', ]
    
    plt.subplot(1,4,1)
    TITLE = "All\npathologists"
    plt.subplots_adjust(wspace = .03)
    fig_df= pd.DataFrame(data_all,columns=[' RS1','RS2','RS3 '])
    bplot=sns.boxplot(data=fig_df,width=0.8,notch=False)
    for j in range(0,3):
        mybox = bplot.artists[j]
        mybox.set_facecolor(continent_colors[j])
    bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
    plt.ylabel(ylabelname,fontsize=FONTSIZE)
    plt.xlabel(xlabelname,fontsize=FONTSIZE)
    ax=plt.gca()
    ax.set_title(TITLE,fontsize=FONTSIZE,pad=-1)
    plt.ylim(YLIM)
    plt.tick_params(labelsize=FONTSIZE)
    
    
    plt.subplot(1,4,2)
    TITLE = "Senior\npathologists"
    plt.subplots_adjust(wspace = .03)
    fig_df= pd.DataFrame(data_H,columns=[' RS1','RS2','RS3 '])
    bplot=sns.boxplot(data=fig_df,width=0.8,notch=False)
    for j in range(0,3):
        mybox = bplot.artists[j]
        mybox.set_facecolor(continent_colors[j])
    bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
    plt.ylabel(ylabelname,fontsize=FONTSIZE)
    plt.xlabel(xlabelname,fontsize=FONTSIZE)
    ax=plt.gca()
    ax.set_title(TITLE,fontsize=FONTSIZE,pad=-1)
    plt.ylim(YLIM)
    plt.tick_params(labelsize=FONTSIZE)
    ax.axes.get_yaxis().set_visible(False)
    
    plt.subplot(1,4,3)
    TITLE = "Intermediate\npathologists"
    plt.subplots_adjust(wspace = .03)
    fig_df= pd.DataFrame(data_M,columns=[' RS1','RS2','RS3 '])
    bplot=sns.boxplot(data=fig_df,width=0.8,notch=False)
    for j in range(0,3):
        mybox = bplot.artists[j]
        mybox.set_facecolor(continent_colors[j])
    bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
    plt.ylabel(ylabelname,fontsize=FONTSIZE)
    plt.xlabel(xlabelname,fontsize=FONTSIZE)
    ax=plt.gca()
    ax.set_title(TITLE,fontsize=FONTSIZE,pad=-1)
    plt.ylim(YLIM)
    plt.tick_params(labelsize=FONTSIZE)
    ax.axes.get_yaxis().set_visible(False)
    
    plt.subplot(1,4,4)
    TITLE = "Junior\npathologists"
    plt.subplots_adjust(wspace = .03)
    fig_df= pd.DataFrame(data_P,columns=[' RS1','RS2','RS3 '])
    bplot=sns.boxplot(data=fig_df,width=0.8,notch=False)
    for j in range(0,3):
        mybox = bplot.artists[j]
        mybox.set_facecolor(continent_colors[j])
    bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
    plt.ylabel(ylabelname,fontsize=FONTSIZE)
    plt.xlabel(xlabelname,fontsize=FONTSIZE)
    ax=plt.gca()
    ax.set_title(TITLE,fontsize=FONTSIZE,pad=-1)
    plt.ylim(YLIM)
    plt.tick_params(labelsize=FONTSIZE)
    ax.axes.get_yaxis().set_visible(False)
    
    
    fig.text(0.5,0.02,'Ring studies',ha='center',va='center',fontsize=FONTSIZE)
    fig.savefig(os.path.join(save_dir,'Fig4_Boxplot_Accuracy_C'+str(CLASS)+'_all_group.pdf'),dpi=300)
    fig.savefig(os.path.join(save_dir,'Fig4_Boxplot_Accuracy_C'+str(CLASS)+'_all_group.png'),dpi=300)


