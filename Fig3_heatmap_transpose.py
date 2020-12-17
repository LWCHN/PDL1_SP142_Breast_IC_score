# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:46:28 2020
@author: leonlwang
"""
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pdb
from datetime import datetime
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y_%b_%d_%Hh%M")
print('Current Timestamp : ', timestampStr)

from utils.tools_pdl1_exp import resize_label_img
from utils.tools_pdl1_exp import IC_class_to_group
from matplotlib.colors import LinearSegmentedColormap
## https://matplotlib.org/3.1.1/gallery/color/custom_cmap.html
## https://www.cnblogs.com/tsingke/p/6218313.html
    
def fig_detail_ic_score_all_user(figname,my_colormap,data_table_array,MEDICINE_NBR,AI_NBR,remove_user_id,EXP):
    user_namelist_all=list(range(1,MEDICINE_NBR+1))
    AMP_IMG=10
    array_data_show_10x = resize_label_img(data_table_array,AMP_IMG*data_table_array.shape[0],AMP_IMG*data_table_array.shape[1])
    
    ## make user_list
    USER_NAME=list()
    for i in range(0,len(user_namelist_all)):
        USER_NAME.append(''+str(user_namelist_all[i]))
    for kk in range(0,AI_NBR):
        if kk==0:
            USER_NAME.append('surfV2')
        if kk==1:
            USER_NAME.append('surfV1')
        if kk==2:
            USER_NAME.append('cell')
    
    IMG_NAME=list()
    for i in range(0,data_table_array.shape[0]):
        if i+1 not in remove_user_id:
            IMG_NAME.append(''+str(i+1))
    
    ## fig1a overlay VALUE
    plt.close('all')
    fig1a, ax = plt.subplots()
    fig1a.set_size_inches(24, 8)
    array_data_show_10x = array_data_show_10x.transpose()
    ax.imshow(array_data_show_10x,cmap=my_colormap)

    # We want to show all ticks...
    ax.set_yticks(np.arange(int(AMP_IMG/2),AMP_IMG*len(USER_NAME),AMP_IMG))
    ax.set_xticks(np.arange(int(AMP_IMG/2),AMP_IMG*len(IMG_NAME),AMP_IMG))
    # ... and label them with the respective list entries
    ax.set_yticklabels(USER_NAME,fontsize=10)
    ax.set_xticklabels(IMG_NAME,fontsize=10)
    plt.xlabel('Image number\nRS'+str(EXP), fontsize=16)
    plt.ylabel('Pathologist ID', fontsize=16)
    for i in range(0,array_data_show_10x.shape[0],AMP_IMG):
        for j in range(0,array_data_show_10x.shape[1],AMP_IMG):
            ax.text(j+int(AMP_IMG/2), i+int(AMP_IMG/2), int(array_data_show_10x[i, j]),ha="center", va="center", color="k",fontsize=12)
    
    fig1a.tight_layout()
    fig1a.savefig(figname+'.pdf',dpi=150)
    plt.close('all')




nbr_color=100
c1_list=[255.0]
c1_list.extend(np.linspace(255.0, 0.0, num=5)[1:])
c1_list.extend(np.linspace(0.0, 97.0, num=6)[1:])
c1_list.extend(np.linspace(97.0, 254.0, num=202)[-90:])

c2_list=[71.0]
c2_list.extend(np.linspace(71.0, 186.0, num=5)[1:])
c2_list.extend(np.linspace(186.0, 156.0, num=6)[1:])
c2_list.extend(np.linspace(156.0, 222.0, num=202)[-90:])

c3_list=[76.0]
c3_list.extend(np.linspace(76.0, 56.0, num=5)[1:])
c3_list.extend(np.linspace(56.0, 255.0, num=6)[1:])
c3_list.extend(np.linspace(255.0, 62.0, num=202)[-90:])


colors_100=list()
for i in range(0,len(c1_list)):
    colors_100.append([c1_list[i],c2_list[i],c3_list[i]])
 
my_rgb1=[(255,71,76),
         (0,186,56),
         (97,156,255),
         (254,222,62)]

colors_100= (np.array(colors_100)/255).tolist()
n_bin = 100  # Discretizes the interpolation into bins
mycmap_c100 = LinearSegmentedColormap.from_list('mycmap_c100', colors_100, N=n_bin)
colors_4=my_rgb1
n_bin = 4  # Discretizes the interpolation into bins
colors_4 = (np.array(colors_4)/255).tolist()
mycmap_c4 = LinearSegmentedColormap.from_list('mycmap_c4', colors_4, N=n_bin)

colors_2=my_rgb1
colors_2 = (np.array(colors_2)/255).tolist()
n_bin = 2
mycmap_c2 = LinearSegmentedColormap.from_list('mycmap_c2', colors_2, N=n_bin)

###############################################################################
## config parameters
###############################################################################
from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman" 

MEDICINE_NBR   = 31
AI_NBR         = 0
remove_user_id = []

user_namelist_high = list(range(1,12))
user_namelist_mid  = list(range(12,22))
user_namelist_low  = list(range(22,MEDICINE_NBR+1))
user_namelist_all  = list(range(1,MEDICINE_NBR+1))
cutoff_4class = [0,1,5,10]
cutoff_2class = [0,1]

save_path='./result_Fig'
img_list_in_order=(np.load('./IC_Score_Pathologist/img_list_in_order.npy')).tolist()    
ic_dict_medicine_exp1 = json.load(open("./IC_Score_Pathologist/ic_dict_medicine_exp1.json",'r'))
ic_dict_medicine_exp2 = json.load(open("./IC_Score_Pathologist/ic_dict_medicine_exp2.json",'r'))
ic_dict_medicine_exp3 = json.load(open("./IC_Score_Pathologist/ic_dict_medicine_exp3.json",'r'))

###############################################################################
## make array for all the computes
###############################################################################
MEDICINE_NBR=MEDICINE_NBR
AI_NBR=AI_NBR
ic_result_continue_exp1=np.zeros([len(img_list_in_order),MEDICINE_NBR+AI_NBR])
ic_result_continue_exp2=np.zeros([len(img_list_in_order),MEDICINE_NBR+AI_NBR])
ic_result_continue_exp3=np.zeros([len(img_list_in_order),MEDICINE_NBR+AI_NBR])

for i in range(0,len(img_list_in_order)):
    img_name=img_list_in_order[i]
    one_row_medicine=np.array(ic_dict_medicine_exp1[img_name])
    ic_result_continue_exp1[i,0:len(one_row_medicine)] = one_row_medicine[:]

    one_row_medicine=np.array(ic_dict_medicine_exp2[img_name])
    ic_result_continue_exp2[i,0:len(one_row_medicine)] = one_row_medicine[:]

    one_row_medicine=np.array(ic_dict_medicine_exp3[img_name])
    ic_result_continue_exp3[i,0:len(one_row_medicine)] = one_row_medicine[:]
    
print('ic_result_continue_exp1.shape = ',ic_result_continue_exp1.shape)
print('ic_result_continue_exp2.shape = ',ic_result_continue_exp2.shape)
print('ic_result_continue_exp3.shape = ',ic_result_continue_exp3.shape)

### make 4- and 2-categary score table
cutoff_4class=[0,1,5,10]
cutoff_2class=[0,1]
ic_reslut_4class_exp1 = IC_class_to_group(ic_result_continue_exp1,cutoff=cutoff_4class)
ic_reslut_4class_exp2 = IC_class_to_group(ic_result_continue_exp2,cutoff=cutoff_4class)
ic_reslut_4class_exp3 = IC_class_to_group(ic_result_continue_exp3,cutoff=cutoff_4class)

ic_reslut_2class_exp1 = IC_class_to_group(ic_result_continue_exp1,cutoff=cutoff_2class)
ic_reslut_2class_exp2 = IC_class_to_group(ic_result_continue_exp2,cutoff=cutoff_2class)
ic_reslut_2class_exp3 = IC_class_to_group(ic_result_continue_exp3,cutoff=cutoff_2class)

    
###############################################################################
## make figures
###############################################################################
print('make figure for exp 1')
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_continue_RS1'),  mycmap_c100, ic_result_continue_exp1, MEDICINE_NBR,AI_NBR,remove_user_id,1)
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_4category_RS1'), mycmap_c4,   ic_reslut_4class_exp1,   MEDICINE_NBR,AI_NBR,remove_user_id,1)
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_2category_RS1'), mycmap_c2,   ic_reslut_2class_exp1,   MEDICINE_NBR,AI_NBR,remove_user_id,1)


print('make figure for exp 2')
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_continue_RS2'),  mycmap_c100, ic_result_continue_exp2, MEDICINE_NBR,AI_NBR,remove_user_id,2)
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_4category_RS2'), mycmap_c4,   ic_reslut_4class_exp2,   MEDICINE_NBR,AI_NBR,remove_user_id,2)
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_2category_RS2'), mycmap_c2,   ic_reslut_2class_exp2,   MEDICINE_NBR,AI_NBR,remove_user_id,2)

print('make figure for exp 3')
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_continue_RS3'),  mycmap_c100, ic_result_continue_exp3, MEDICINE_NBR,AI_NBR,remove_user_id,3)
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_4category_RS3'), mycmap_c4,   ic_reslut_4class_exp3,   MEDICINE_NBR,AI_NBR,remove_user_id,3)
fig_detail_ic_score_all_user(os.path.join(save_path,'Fig3_IC_2category_RS3'), mycmap_c2,   ic_reslut_2class_exp3,   MEDICINE_NBR,AI_NBR,remove_user_id,3)

