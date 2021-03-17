# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:46:28 2020
@author: leonlwang
"""

import sys
import os
import pandas as pd
import numpy as np
#from nipype.algorithms.icc import ICC_rep_anova
import json
import matplotlib.pyplot as plt
from PIL import Image
import cv2


from datetime import datetime
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y_%b_%d_%Hh%M")
print('Current Timestamp : ', timestampStr)

sys.path.insert(0, './')
sys.path.insert(1, './utils')
from tools_pdl1_exp import json_to_array,save_df_to_xlsx
from icc_util import calc_icc_ci95,show_only_icc31,make_array_from_userlist

from tools_pdl1_exp import IC_class_to_group
from icc_util import calc_icc_ci95,show_only_icc31,calc_Fleiss_kappa_subgroup,make_array_from_userlist

from icc_util import calc_icc21_array
from tools_pdl1_exp import save_icc21_array_to_excel

# ###############################################################################
# ## config parameters
# ###############################################################################
MEDICINE_NBR=31
user_namelist_high=list(range(1,12))
user_namelist_mid=list(range(12,22))
user_namelist_low=list(range(22,MEDICINE_NBR+1))
data_path = './IC_Score_Pathologist'
save_path = './result_Calc'
if not os.path.exists(save_path): os.mkdir(save_path)

data_table_exp1 = json_to_array(os.path.join(data_path,'img_list_in_order.npy'),os.path.join(data_path,"ic_dict_medicine_exp1.json") )
data_table_exp2 = json_to_array(os.path.join(data_path,'img_list_in_order.npy'),os.path.join(data_path,"ic_dict_medicine_exp2.json") )
data_table_exp3 = json_to_array(os.path.join(data_path,'img_list_in_order.npy'),os.path.join(data_path,"ic_dict_medicine_exp3.json") )

### make Class_array in order
cutoff_4class=[0,1,5,10]
cutoff_2class=[0,1]
ic_reslut_4class_exp1 = IC_class_to_group(data_table_exp1,cutoff=cutoff_4class)
ic_reslut_4class_exp2 = IC_class_to_group(data_table_exp2,cutoff=cutoff_4class)
ic_reslut_4class_exp3 = IC_class_to_group(data_table_exp3,cutoff=cutoff_4class)

ic_reslut_2class_exp1 = IC_class_to_group(data_table_exp1,cutoff=cutoff_2class)
ic_reslut_2class_exp2 = IC_class_to_group(data_table_exp2,cutoff=cutoff_2class)
ic_reslut_2class_exp3 = IC_class_to_group(data_table_exp3,cutoff=cutoff_2class)
    
if 1:
    #########################
    #### ICC 31 for all user
    #########################
    df_icc_exp1 = calc_icc_ci95(data_table_exp1)
    save_df_to_xlsx(df_icc_exp1, os.path.join(save_path ,"ICC_ci95_exp1_all.xlsx"))
    df_icc_exp2 = calc_icc_ci95(data_table_exp2)
    save_df_to_xlsx(df_icc_exp2, os.path.join(save_path ,"ICC_ci95_exp2_all.xlsx"))
    df_icc_exp3 = calc_icc_ci95(data_table_exp3)
    save_df_to_xlsx(df_icc_exp3, os.path.join(save_path ,"ICC_ci95_exp3_all.xlsx"))
    print('df_icc_exp1 = ',df_icc_exp1)
    print('df_icc_exp2 = ',df_icc_exp2)
    print('df_icc_exp3 = ',df_icc_exp3)
    show_only_icc31(df_icc_exp1,df_icc_exp2,df_icc_exp3,"ICC31_CI95_exp123 (col=Exp1, Exp2,Exp3; row=ICC, CI95-up, CI95-low) =")
        
    #########################
    #### ICC 31 for 3 levels of pathologist
    #########################
    ## Exp1: Senior, Intermediate,Junior
    df_icc_exp1_high = calc_icc_ci95(make_array_from_userlist(data_table_exp1,user_namelist_high)) 
    df_icc_exp1_mid  = calc_icc_ci95(make_array_from_userlist(data_table_exp1,user_namelist_mid))
    df_icc_exp1_low   = calc_icc_ci95(make_array_from_userlist(data_table_exp1,user_namelist_low))
    save_df_to_xlsx(df_icc_exp1_high, os.path.join(save_path ,"ICC31_CI95_exp1_senior.xlsx"))
    save_df_to_xlsx(df_icc_exp1_mid, os.path.join(save_path ,"ICC31_CI95_exp1_intermediate.xlsx"))
    save_df_to_xlsx(df_icc_exp1_low, os.path.join(save_path ,"ICC31_CI95_exp1_junior.xlsx"))
    print('\n\n\n ---- Exp1 ----')
    show_only_icc31(df_icc_exp1_high,df_icc_exp1_mid,df_icc_exp1_low, "ICC31_CI95_exp1 (3 levels)  (col=Senior, Intermediate,Junior; row=icc, ci95up,ci95low) =")

    ## Exp2: Senior, Intermediate,Junior
    df_icc_exp2_high = calc_icc_ci95(make_array_from_userlist(data_table_exp2,user_namelist_high)) 
    df_icc_exp2_mid = calc_icc_ci95(make_array_from_userlist(data_table_exp2,user_namelist_mid))
    df_icc_exp2_low = calc_icc_ci95(make_array_from_userlist(data_table_exp2,user_namelist_low))
    save_df_to_xlsx(df_icc_exp2_high, os.path.join(save_path ,"ICC31_CI95__exp2_senior.xlsx"))
    save_df_to_xlsx(df_icc_exp2_mid, os.path.join(save_path ,"ICC31_CI95__exp2_intermediate.xlsx"))
    save_df_to_xlsx(df_icc_exp2_low, os.path.join(save_path ,"ICC31_CI95__exp2_junior.xlsx"))
    print('\n\n\n ---- Exp2 ----')
    show_only_icc31(df_icc_exp2_high,df_icc_exp2_mid,df_icc_exp2_low, "ICC31_CI95_exp2 (3 levels)  (col=Senior, Intermediate,Junior; row=icc, ci95up,ci95low) =")

    ## Exp3: Senior, Intermediate,Junior
    df_icc_exp3_high = calc_icc_ci95(make_array_from_userlist(data_table_exp3,user_namelist_high)) 
    df_icc_exp3_mid = calc_icc_ci95(make_array_from_userlist(data_table_exp3,user_namelist_mid))
    df_icc_exp3_low = calc_icc_ci95(make_array_from_userlist(data_table_exp3,user_namelist_low))
    save_df_to_xlsx(df_icc_exp3_high, os.path.join(save_path ,"ICC31_CI95__exp3_senior.xlsx"))
    save_df_to_xlsx(df_icc_exp3_mid, os.path.join(save_path ,"ICC31_CI95__exp3_intermediate.xlsx"))
    save_df_to_xlsx(df_icc_exp3_low, os.path.join(save_path ,"ICC31_CI95__exp3_junior.xlsx"))
    print('\n\n\n ---- Exp3 ----')
    show_only_icc31(df_icc_exp3_high,df_icc_exp3_mid,df_icc_exp3_low, "ICC31_CI95_exp3 (3 levels)  (col=Senior, Intermediate,Junior; row=icc, ci95up,ci95low) =")


if 1:
    #########################
    #### Fleiss Kappa ###########
    #########################
    FK_2class_exp1 = calc_Fleiss_kappa_subgroup(ic_reslut_2class_exp1,user_namelist_high,user_namelist_mid,user_namelist_low)
    FK_2class_exp2 = calc_Fleiss_kappa_subgroup(ic_reslut_2class_exp2,user_namelist_high,user_namelist_mid,user_namelist_low)
    FK_2class_exp3 = calc_Fleiss_kappa_subgroup(ic_reslut_2class_exp3,user_namelist_high,user_namelist_mid,user_namelist_low)

    FK_4class_exp1 = calc_Fleiss_kappa_subgroup(ic_reslut_4class_exp1,user_namelist_high,user_namelist_mid,user_namelist_low)
    FK_4class_exp2 = calc_Fleiss_kappa_subgroup(ic_reslut_4class_exp2,user_namelist_high,user_namelist_mid,user_namelist_low)
    FK_4class_exp3 = calc_Fleiss_kappa_subgroup(ic_reslut_4class_exp3,user_namelist_high,user_namelist_mid,user_namelist_low)
    print("FK_2class_exp1,FK_2class_exp2,FK_2class_exp3 \n [total, high, mid,low] = \n",FK_2class_exp1, '\n', FK_2class_exp2, '\n', FK_2class_exp3)
    print("FK_4class_exp1,FK_4class_exp2,FK_4class_exp3  \n [total, high, mid,low] = \n",FK_4class_exp1, '\n', FK_4class_exp2, '\n', FK_4class_exp3)
    
    
if 1:
    #########################
    #### ICC 21 for 3 levels of pathologist
    #########################
    icc21_exp12 = calc_icc21_array(data_table_exp1,data_table_exp2)
    save_icc21_array_to_excel(os.path.join(save_path ,"ICC21_CI95_exp12.xlsx"),icc21_exp12)

    icc21_exp13 = calc_icc21_array(data_table_exp1,data_table_exp3)
    save_icc21_array_to_excel(os.path.join(save_path ,"ICC21_CI95_exp13.xlsx"),icc21_exp13)

    icc21_exp23 = calc_icc21_array(data_table_exp2,data_table_exp3)
    save_icc21_array_to_excel(os.path.join(save_path ,"ICC21_CI95_exp23.xlsx"),icc21_exp23)
   
    icc21_exp12_2class = calc_icc21_array(ic_reslut_2class_exp1,ic_reslut_2class_exp2)
    save_icc21_array_to_excel(os.path.join(save_path ,"ICC21_CI95_exp12_2class.xlsx"),icc21_exp12_2class)
    
if 1:
    #####################
    ## for one user: how many different scoring on same image from RS1, RS2
    #####################
    nbr_of_diff_each_user=np.zeros([1,31])
    for user_index in range(0,ic_reslut_2class_exp1.shape[1]):
        data_rs1=ic_reslut_2class_exp1[:,user_index].astype(np.uint8)
        data_rs2=ic_reslut_2class_exp2[:,user_index].astype(np.uint8)
        compare_result=np.logical_xor(data_rs1,data_rs2).astype(np.uint8)
        nbr_of_diff_each_user[0,user_index]=compare_result.sum()
    
    
    np.savetxt(os.path.join(save_path,'nbr_of_diff_from_each_user_2class_between_RS1_RS2.out'), nbr_of_diff_each_user,fmt='%d')
    print('nbr_of_diff_from_each_user_2class_between_RS1_RS2 = \n',nbr_of_diff_each_user)
    print(nbr_of_diff_each_user/109)
    # plt.figure()
    # plt.plot(nbr_of_diff_each_user[0,:],'ro-')



