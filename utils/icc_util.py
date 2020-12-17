# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:55:32 2020

@author: leonlwang
"""
import numpy as np
import pandas as pd
# icc 31
import pingouin as pg
pg.options['round.column.CI95%'] = 5
pg.options['round.column.ICC'] = 5
pg.options['round'] = None

#  Fleiss Kappa
from statsmodels.stats.inter_rater import fleiss_kappa
import scipy

####################
######  ICC31  ########
####################
def convert_array_to_targe_rater_rating(array2D):
    ## row = targe 
    ## col = rater
    ## convert array [col, row] into  [col*row, 3]
    array_for_pingouin=np.zeros([array2D.shape[0]*array2D.shape[1],3])
    record_index=0
    for row in range(0,array2D.shape[0]):
        for col in range(0,array2D.shape[1]):
            targe_index = row
            rater_index = col   
            value  = array2D[row,col]
            array_for_pingouin[record_index,:] = np.array([targe_index,rater_index,value])
            record_index=record_index+1
    return array_for_pingouin
    
def get_array_df_pingouin(one_medicine):
    array_for_pingouin = convert_array_to_targe_rater_rating(one_medicine)
    return pd.DataFrame(array_for_pingouin,columns=['target','rater','rating'])
    
 ## for  ICC(3,1)
def calc_icc_ci95(data_table_exp):
    df_array_exp= get_array_df_pingouin(data_table_exp)
    icc_exp = pg.intraclass_corr(df_array_exp, targets='target', raters='rater', ratings='rating').round(4)
    return icc_exp

def show_only_icc31(df_icc_exp1,df_icc_exp2,df_icc_exp3, Description):
    ICC31_exp123      = np.array([ df_icc_exp1['ICC'][2],      df_icc_exp2['ICC'][2],      df_icc_exp3['ICC'][2]       ])
    ICC31_CI95_Low    = np.array([ df_icc_exp1['CI95%'][2][0], df_icc_exp2['CI95%'][2][0], df_icc_exp3['CI95%'][2][0]  ])
    ICC31_CI95_Up     = np.array([ df_icc_exp1['CI95%'][2][1], df_icc_exp2['CI95%'][2][1], df_icc_exp3['CI95%'][2][1]  ])
    ICC31_CI95_exp123 = np.array([ ICC31_exp123,ICC31_CI95_Low,ICC31_CI95_Up])
    print(Description,'\n',ICC31_CI95_exp123)
    
    
########################
###### Fleiss Kappa  ########
########################
def calc_Fleiss_kappa(ic_reslut_Nclass):
    categories=list(set(list(  ic_reslut_Nclass.flatten() )))
    ic_reslut_Nclass.astype(np.uint8)
    
    array_for_Fkappa=np.zeros([ic_reslut_Nclass.shape[0], len(categories)])
    array_for_Fkappa.astype(np.uint8)
    for row in range(0,ic_reslut_Nclass.shape[0]):
        for col in range(0,ic_reslut_Nclass.shape[1]):
            for class_value in categories:
                class_value=int(class_value)
                if int(ic_reslut_Nclass[row,col])==int(class_value):
                    array_for_Fkappa[row,class_value]=array_for_Fkappa[row,class_value]+1
    array_for_Fkappa.astype(np.uint8)
    ##print(array_for_Fkappa)
    return fleiss_kappa(array_for_Fkappa)

def make_array_from_userlist(data_table_array_all,user_namelist):
    output_array=np.zeros([data_table_array_all.shape[0], len(user_namelist)])
    column_index=0
    for user_id in user_namelist:
        #print('user_id=',user_id)
        output_array[:,column_index]=data_table_array_all[:,user_id-1]
        column_index=column_index+1
    return output_array
    
def calc_Fleiss_kappa_subgroup(ic_reslut_by_class,user_namelist_high,user_namelist_mid,user_namelist_low):
    total     = calc_Fleiss_kappa(ic_reslut_by_class)
    high = calc_Fleiss_kappa(make_array_from_userlist(ic_reslut_by_class,user_namelist_high))
    mid  = calc_Fleiss_kappa(make_array_from_userlist(ic_reslut_by_class,user_namelist_mid))
    low  = calc_Fleiss_kappa(make_array_from_userlist(ic_reslut_by_class,user_namelist_low))
    print('FleissK expM Nclass: total, high, mid,low', np.round( np.array([total, high,mid, low]) , decimals=3)    )
    return np.round(np.array([total, high,mid, low]) ,decimals=5)


def mean_confidence_interval(data, confidence=0.95):
    ### https://www.it-swarm.dev/zh/python/%E4%BB%8E%E6%A0%B7%E6%9C%AC%E6%95%B0%E6%8D%AE%E8%AE%A1%E7%AE%97%E7%BD%AE%E4%BF%A1%E5%8C%BA%E9%97%B4/1071261127/
    ### https://www.statisticshowto.com/tables/t-distribution-table/#two
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
    
    
#########################
#### ICC 21 ###############
#########################
def calc_icc21_array(data_table_exp1,data_table_exp2):
    icc_bt_exp=np.zeros([3,data_table_exp1.shape[1]])
    for i in range(0,data_table_exp1.shape[1]):
    #    print('i=',i,'_'*30)    
        one_medicine=np.zeros([data_table_exp1.shape[0],2])
        one_medicine[:,0]=data_table_exp1[:,i]
        one_medicine[:,1]=data_table_exp2[:,i]
    #    icc_all_para=ICC_rep_anova(one_medicine)
    #    print('icc_all_para[0] = icc31 = ',np.round(icc_all_para[0],decimals=3))
        df_one_medicine_exp12 = get_array_df_pingouin(one_medicine)
        icc_one_medicine_exp12 = pg.intraclass_corr(df_one_medicine_exp12, targets='target', raters='rater', ratings='rating')
        ICC31=icc_one_medicine_exp12['ICC'][2]
        ICC31_CI95=icc_one_medicine_exp12['CI95%'][2]
        ICC21=icc_one_medicine_exp12['ICC'][1]
        ICC21_CI95=icc_one_medicine_exp12['CI95%'][1]
    #    print(icc_one_medicine_exp12)
        
        icc_bt_exp[0,i] = ICC21
        icc_bt_exp[1,i] = ICC21_CI95[0]
        icc_bt_exp[2,i] = ICC21_CI95[1]
    return  icc_bt_exp

