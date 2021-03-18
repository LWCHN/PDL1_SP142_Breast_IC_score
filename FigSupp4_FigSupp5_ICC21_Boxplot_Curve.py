# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:44:47 2020
@author: leonlwang
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

def fig_boxplot_icc21(icc12_exp12,save_dir,save_name):
    ## [2] prepare data
    icc21 = icc12_exp12[0,:]
    ci95low=icc12_exp12[1,:]
    ci95up=icc12_exp12[2,:]
    
    icc21_H = icc12_exp12[0,0:11]
    ci95low_H=icc12_exp12[1,0:11]
    ci95up_H=icc12_exp12[2,0:11]
    
    icc21_M = icc12_exp12[0,11:21]
    ci95low_M=icc12_exp12[1,11:21]
    ci95up_M=icc12_exp12[2,11:21]
    
    icc21_P = icc12_exp12[0,21:31]
    ci95low_P=icc12_exp12[1,21:31]
    ci95up_P=icc12_exp12[2,21:31]
    
    ## [3] figure
    ylabelname='ylabelname'
    xlabelname='xlabelname'
    plt.close('all')
    title=['All', 'Senior','Intermediate','Junior']
    fig=plt.figure()
    fig.set_size_inches(6, 4)
    FONTSIZE=12
    fig.suptitle('ICC21 for different groups',fontsize=FONTSIZE)
    continent_colors=["w","w","w"]
    YLIM=[0,1.0]
    for i in range(0,4):
        plt.subplot(1,4,i+1)
        plt.axhspan(0, 0.5, facecolor='0.3', alpha=0.5)
        plt.axhspan(0.5, 0.75, facecolor='0.5', alpha=0.5)
        plt.axhspan(0.75, 0.9, facecolor='0.7', alpha=0.5)
        plt.axhspan(0.9, 1.0, facecolor='0.9', alpha=0.5)
        if i==0:
            fig_array=icc21.transpose()
            fig_df= pd.DataFrame(fig_array,columns=[title[i]+'\n'+'pathologists'])
    
            bplot=sns.boxplot(data=fig_df,width=0.6)
            for j in range(0,1):
                mybox = bplot.artists[j]
                mybox.set_facecolor(continent_colors[j])
            bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
            plt.tick_params(labelsize=FONTSIZE)
            ax=plt.gca()
            ax.spines['right'].set_color('none')
            plt.ylim(YLIM)
        elif i==1:
            fig_array=icc21_H.transpose()
            fig_df= pd.DataFrame(fig_array,columns=[title[i]+'\n'+'pathologists'])
            bplot=sns.boxplot(data=fig_df,width=0.6)
            for j in range(0,1):
                mybox = bplot.artists[j]
                mybox.set_facecolor(continent_colors[j])
            bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
            plt.tick_params(labelsize=FONTSIZE)
            ax=plt.gca()
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['right'].set_color('green')
            ax.spines['left'].set_color('none')
            plt.ylim(YLIM)
        elif i==2:
            fig_array=icc21_M.transpose()
            fig_df= pd.DataFrame(fig_array,columns=[title[i]+'\n'+'pathologists'])
            bplot=sns.boxplot(data=fig_df,width=0.6)
            for j in range(0,1):
                mybox = bplot.artists[j]
                mybox.set_facecolor(continent_colors[j])
            bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
            plt.tick_params(labelsize=FONTSIZE)
            ax=plt.gca()
            ax.axes.get_yaxis().set_visible(False)
            ax.spines['right'].set_color('none')
            ax.spines['left'].set_color('none')
            plt.ylim(YLIM)
        elif i==3:
            fig_array=icc21_P.transpose()
            fig_df= pd.DataFrame(fig_array,columns=[title[i]+'\n'+'pathologists'])
            bplot=sns.boxplot(data=fig_df,width=0.6)
            for j in range(0,1):
                mybox = bplot.artists[j]
                mybox.set_facecolor(continent_colors[j])
            bplot = sns.stripplot(data=fig_df,jitter=True, marker='o',alpha=0.7, color="black")
            plt.tick_params(labelsize=FONTSIZE)
            ax=plt.gca()
            ax.yaxis.tick_right()
            ax.set_yticks([0.25,0.625,0.825,0.95])
            ax.set_yticklabels(['Poor','Moderate','Good','Excellent\n\n'])
            plt.yticks(rotation=-90, va='center',ha='center',fontsize=FONTSIZE)
            ax.tick_params(axis='y', length=0,pad=5)
            ax.spines['left'].set_color('none')
            plt.ylim(YLIM)
    
    fig.tight_layout(w_pad=-1)
    fig.savefig(os.path.join(result_Fig_dir ,save_name+'.pdf'),dpi=400)
    fig.savefig(os.path.join(result_Fig_dir ,save_name+'.png'),dpi=400)
    
def print_ave_std_for_paper(array, comments=''):
    print('ICC [ave, std] = ',array[0,:].mean(), array[0,:].std())
    print('CI95_low [ave, std] = ',array[1,:].mean(), array[1,:].std())
    print('CI95_high [ave, std] = ',array[2,:].mean(), array[2,:].std())


def fig_icc21_curve(icc12_exp12,save_dir,save_name,linecolor='r'):
    FONTSIZE=16
    plt.close('all')
    fig, ax = plt.subplots()
    fig.set_size_inches(11.69, 4.5) # 11.69 is A4 paper width
    loc_x=range(0,31)
    labels_x=range(1,32)
    ax.plot(loc_x,icc12_exp12[0,:],color=linecolor, marker='o', linestyle='dashed',markerfacecolor='white')
    ax.fill_between(loc_x,(icc12_exp12[1,:]),(icc12_exp12[2,:]),color=linecolor, alpha=.2)
    
    plt.xticks(loc_x, labels_x)
    plt.xlabel('Pathologist ID',fontsize=FONTSIZE)
    plt.title('Single pathologist absolute agreement ICC21 and 95% CI',fontsize=FONTSIZE)
    plt.grid(linestyle="--", alpha=0.6)
    ax.grid(True, which='both')
    plt.tick_params(labelsize=FONTSIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,save_name+'.pdf'),dpi=400)
    fig.savefig(os.path.join(save_dir,save_name+'.png'),dpi=400)
    
    H = icc12_exp12[0,0:11]
    M = icc12_exp12[0,11:21]
    P = icc12_exp12[0,21:31]
    print(np.round( np.array([H.mean(),M.mean(),P.mean()]),decimals=3) )
    print(np.round( np.array([H.std(),M.std(),P.std()]),decimals=3) )


# [1] FigSupp4a: show only the number of diff on 2 class
def fig_nbr_of_diff_each_user_curve(icc21_exp12,save_dir,save_name,linecolor='r'):
    FONTSIZE=16
    plt.close('all')
    fig, ax = plt.subplots()
    fig.set_size_inches(11.69, 4.5) # 11.69 is A4 paper width
    loc_x=range(0,31)
    labels_x=range(1,32)
    ax.plot(loc_x,icc21_exp12,color=linecolor, marker='^', linestyle='solid',markerfacecolor=linecolor)
    
    plt.xticks(loc_x, labels_x)
    plt.xlabel('Pathologist ID',fontsize=FONTSIZE)
    plt.title('Number of inconsistent binary diagnosis between RS1 and RS2 of each pathologist',fontsize=FONTSIZE)
    plt.grid(linestyle="--", alpha=0.6)
    ax.grid(True, which='both')
    plt.tick_params(labelsize=FONTSIZE)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir,save_name+'.pdf'),dpi=400)
    fig.savefig(os.path.join(save_dir,save_name+'.png'),dpi=400)
    
    H = icc21_exp12[0:11]
    M = icc21_exp12[11:21]
    P = icc21_exp12[21:31]
    print('________ Mean, Std of ICC21 for 3 LEVEL ________')
    print(np.round( np.array([icc21_exp12.mean(), H.mean(),M.mean(),P.mean()]),decimals=3) )
    print(np.round( np.array([icc21_exp12.std(), H.std(),M.std(),P.std()]),decimals=3) )
   





result_Calc_dir = './result_Calc'
result_Fig_dir  = './result_Fig'
#############
##  FigSupp4a
#############
nbr_of_diff_each_user = np.loadtxt(os.path.join(result_Calc_dir,'nbr_of_diff_from_each_user_2class_between_RS1_RS2.out'))
fig_nbr_of_diff_each_user_curve(nbr_of_diff_each_user,result_Fig_dir,'FigSupp4a_nbr_of_diff_each_user',linecolor='k')


#############
##  FigSupp4b
##  FigSupp4c
#############
# [2] see 2CLASS result icc12_exp12
icc21_exp12_2CLASS=np.load(os.path.join(result_Calc_dir,'ICC21_CI95_exp12_2class.xlsx.npy'))
icc21_exp12_2CLASS[ np.isnan(icc21_exp12_2CLASS) == True] = 1.0      
fig_icc21_curve(icc21_exp12_2CLASS,result_Fig_dir,'FigSupp4b_icc21_exp12_with_HMP_curve_2CLASS',linecolor='r')
fig_boxplot_icc21(icc21_exp12_2CLASS,result_Fig_dir,'FigSupp4c_icc21_exp12_with_HMP_boxplot_2CLASS')

#############
##  FigSupp5a
##  FigSupp5b
#############
# [3] figure curve
icc21_exp13=np.load(os.path.join(result_Calc_dir,'ICC21_CI95_exp13.xlsx.npy'))
fig_icc21_curve(icc21_exp13,result_Fig_dir,'FigSupp5a_icc21_exp12_curve',linecolor='g')

icc21_exp23=np.load(os.path.join(result_Calc_dir,'ICC21_CI95_exp23.xlsx.npy'))
fig_icc21_curve(icc21_exp23,result_Fig_dir,'FigSupp5b_icc21_exp12_curve',linecolor='b')