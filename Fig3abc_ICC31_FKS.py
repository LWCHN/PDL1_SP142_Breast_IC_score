# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:01:10 2020
@author: leonlwang
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman" 

def draw_figure_icc(icc31,ci95low,ci95up,TITLE,YLIM):
    plt.close('all')
    title=['All', 'Senior','Intermediate','Junior']
    fig=plt.figure()
    fig.set_size_inches(6, 4)
    FONTSIZE=12
    fig.suptitle(TITLE,fontsize=FONTSIZE)
    fig.subplots_adjust(top=0.85)
    
    for i in range(0,4):
        y = np.array([icc31[i],icc31[i+4],icc31[i+8]])
        x = np.arange(len(y))
        yerr_low=np.array([(icc31-ci95low)[i],(icc31-ci95low)[i+4],(icc31-ci95low)[i+8]])
        yerr_up=np.array([(ci95up-icc31)[i],(ci95up-icc31)[i+4],(ci95up-icc31)[i+8]])
        
        plt.subplot(1,4,i+1)
        plt.subplots_adjust(wspace = .08)
        plt.errorbar(x, y,  yerr=yerr_low, marker='o', mfc='w',mec='k',ms=3, linestyle='', ecolor='r',capsize=0.1,capthick=0, uplims=True, lolims=False)
        plt.errorbar(x, y,  yerr=yerr_up,  marker='o', mfc='w',mec='k',ms=3, linestyle='', ecolor='r',capsize=0.1,capthick=0, uplims=False,lolims=True)
        plt.axhspan(0, 0.5, facecolor='0.3', alpha=0.5)
        plt.axhspan(0.5, 0.75, facecolor='0.5', alpha=0.5)
        plt.axhspan(0.75, 0.9, facecolor='0.7', alpha=0.5)
        plt.axhspan(0.9, 1.0, facecolor='0.9', alpha=0.5)
    
        ax=plt.gca()
        ax.set_title(title[i]+'\n'+'pathologists',fontsize=FONTSIZE)
        ax.set_xticks(np.arange(0,3))
        ax.set_xticklabels(['    RS1','RS2','RS3    '],fontsize=FONTSIZE)
        if i==0:
            plt.yticks(fontsize=FONTSIZE)
        elif i==3:
            ax.yaxis.tick_right()
            ax.set_yticks([0.45,0.625,0.825,0.95])
            ax.set_yticklabels(['Poor','Moderate','Good','Excellent'])
            plt.yticks(rotation=-90, va='center',ha='center',fontsize=FONTSIZE)
            ax.tick_params(axis='y', length=0,pad=10)
        else :
            ax.axes.get_yaxis().set_visible(False)
        plt.ylim(YLIM)
        del ax

    fig.text(0.5,0.020,'Ring studies',ha='center',va='center',fontsize=FONTSIZE)
    return fig



def draw_figure_fks(icc31,ci95low,ci95up,TITLE,YLIM):
    plt.close('all')
    title=['All', 'Senior','Intermediate','Junior']
    fig=plt.figure()
    fig.set_size_inches(6, 4)
    FONTSIZE=12
    
    fig.suptitle(TITLE,fontsize=FONTSIZE)
    fig.subplots_adjust(top=0.85)

    for i in range(0,4):
        y = np.array([icc31[i],icc31[i+4],icc31[i+8]])
        x = np.arange(len(y))
        yerr_low=np.array([(icc31-ci95low)[i],(icc31-ci95low)[i+4],(icc31-ci95low)[i+8]])
        yerr_up=np.array([(ci95up-icc31)[i],(ci95up-icc31)[i+4],(ci95up-icc31)[i+8]])
        
        plt.subplot(1,4,i+1)
        plt.subplots_adjust(wspace = .08)
        plt.errorbar(x, y,  yerr=yerr_low, marker='o', mfc='w',mec='k',ms=3,linestyle='', ecolor='r',capsize=0.1,capthick=0, uplims=True, lolims=False)
        plt.errorbar(x, y,  yerr=yerr_up,  marker='o', mfc='w',mec='k',ms=3,linestyle='', ecolor='r',capsize=0.1,capthick=0, uplims=False,lolims=True)
        plt.axhspan(0, 0.4, facecolor='0.2', alpha=0.5)
        plt.axhspan(0.4, 0.6, facecolor='0.3', alpha=0.5)
        plt.axhspan(0.6, 0.8, facecolor='0.5', alpha=0.5)
        plt.axhspan(0.8, 0.9, facecolor='0.7', alpha=0.5)
        plt.axhspan(0.9, 1.0, facecolor='0.9', alpha=0.5)
    
        ax=plt.gca()
        ax.set_title(title[i]+'\n'+'pathologists',fontsize=FONTSIZE)
        ax.set_xticks(np.arange(0,3))
        ax.set_xticklabels(['    RS1','RS2','RS3    '],fontsize=FONTSIZE)
        if i==0:
            plt.yticks(fontsize=FONTSIZE)
        elif i==3:
            ax.yaxis.tick_right()
            ax.set_yticks([0.2,0.5,0.7,0.85,0.95])
            ax.set_yticklabels(['','\nWeak', '\nModerate', '\nStrong','Near\nperfect'])
            plt.yticks(rotation=-90, va='center',ha='center',fontsize=FONTSIZE)
            ax.tick_params(axis='y', length=0,pad=15)
        else :
            ax.axes.get_yaxis().set_visible(False)
        plt.ylim(YLIM)
        del ax
    
    fig.text(0.5,0.02,'Ring studies',ha='center',va='center',fontsize=FONTSIZE)
    return fig






plt.close('all')
## [1] read from excel
result_Calc_dir='./result_Calc'
result_Fig_dir = './result_Fig'

data = pd.read_excel(os.path.join(result_Calc_dir,'concordance_icc_fks.xlsx'),header =0,index_col=0)
print(data.index)
print(data.columns)

## [2] figure
icc31=np.array(data['icc'])
ci95low=np.array(data['ci_low'])
ci95up=np.array(data['ci_up'])
fig_icc = draw_figure_icc(icc31,ci95low,ci95up,'Concordance: ICC(3,1)\n\n',[0.39,1])
fig_icc.savefig(os.path.join(result_Fig_dir,'Fig3a_ICC31.pdf'),dpi=300)
fig_icc.savefig(os.path.join(result_Fig_dir,'Fig3a_ICC31.png'),dpi=300)
fig_icc.savefig(os.path.join(result_Fig_dir,'Fig3a_ICC31.eps'),format='eps') 


fks_2c=np.array(data['fks_2c'])
ci95low=np.array(data['ci_low_2c'])
ci95up=np.array(data['ci_up_2c'])
fig_fks_2c = draw_figure_fks(fks_2c,ci95low,ci95up,'Concordance: FKS 2 Categories',[0.39,1])
fig_fks_2c.savefig(os.path.join(result_Fig_dir,'Fig3b_FKS_2Class.pdf'),dpi=300)
fig_fks_2c.savefig(os.path.join(result_Fig_dir,'Fig3b_FKS_2Class.png'),dpi=300)
fig_fks_2c.savefig(os.path.join(result_Fig_dir,'Fig3b_FKS_2Class.eps'),format='eps')

fks_4c=np.array(data['fks_4c'])
ci95low=np.array(data['ci_low_4c'])
ci95up=np.array(data['ci_up_4c'])
fig_fks_4c = draw_figure_fks(fks_4c,ci95low,ci95up,'Concordance: FKS 4 Categories',[0.39,1])
fig_fks_4c.savefig(os.path.join(result_Fig_dir,'Fig3c_FKS_4Class.pdf'),dpi=300)
fig_fks_4c.savefig(os.path.join(result_Fig_dir,'Fig3c_FKS_4Class.png'),dpi=300)
fig_fks_4c.savefig(os.path.join(result_Fig_dir,'Fig3c_FKS_4Class.eps'),format='eps')