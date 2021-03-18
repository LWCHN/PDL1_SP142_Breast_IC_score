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
import sklearn
from sklearn import metrics


from utils.metric_util_PDL1 import compute_performance
from utils.tools_pdl1_exp import IC_one_value_to_group


def ic_dict_to_array(dict_gt,dict_pred):
    gt_list=list()
    pred_list=list()
    for key, value in dict_pred.items():
        pred_list.append(value)
        gt_list.append(dict_gt[key])
    return np.array(gt_list), np.array(pred_list)


## calc high, mid. primary
def calc_mean_std(array,decimals=3):
    #print(np.round(np.array([  array.mean(), array.std() ] ),decimals=decimals))
    return np.round(np.array([  array.mean(), array.std() ] ),decimals=decimals)

def calc_total_HMP(result_df,METRIC_NAME):
    total   = calc_mean_std(result_df[METRIC_NAME].to_numpy())
    high    = calc_mean_std(result_df[METRIC_NAME].to_numpy()[0:11])
    mid     = calc_mean_std(result_df[METRIC_NAME].to_numpy()[11:21])
    primary = calc_mean_std(result_df[METRIC_NAME].to_numpy()[21:31])
    print('def calc_total_HMP(): ave +- std')
    print('METRIC_NAME = ',METRIC_NAME)
    print('total,high,mid,primary')
    print(total,high,mid,primary)
    return np.array([total,high,mid,primary])

def calc_all(EXP):
    path_IC_Score_Pathologist = './IC_Score_Pathologist'
    result_Calc = './result_Calc'

    #############
    ## 4 Class ##
    #############
    
    ## [1] load ground truth make ground truth array, with name order
    ic_dict_ground_truth        = json.load(open(os.path.join(path_IC_Score_Pathologist ,"ic_dict_ground_truth.json"),'r'))
    ic_dict_ground_truth_4class = json.load(open(os.path.join(path_IC_Score_Pathologist ,"ic_dict_ground_truth_4class.json"),'r'))
    img_list_in_order=np.load(os.path.join(path_IC_Score_Pathologist ,'img_list_in_order.npy')).tolist()
    
    
    ## [2] load algo, with pre-process 4 clsss, 2 class
    user31_result = json.load(open(os.path.join(path_IC_Score_Pathologist ,"ic_dict_medicine_exp"+str(EXP)+".json"),'r'))
    csv_name  = os.path.join(result_Calc,'evaluate_user31_exp'+str(EXP)+'_4class.csv')
    print("++"*100)
    print(csv_name)
    cutoff_4class=[0,1,5,10]
    
    if os.path.exists(csv_name):
        os.remove(csv_name)
    
    for user_index in range(0,31):
    #    print("_"*100)
    #    print('user_index = ',user_index)
        one_user_result_4class=dict()
        for one_key, one_value in user31_result.items():
            img_ic_score = one_value[user_index]
            one_user_result_4class.update({one_key: IC_one_value_to_group(img_ic_score,cutoff_4class)})
        
        
        ##  [3] #########   evaluation #####################
        ## convert dict 2 list with order
        gt4c,user4c =  ic_dict_to_array(ic_dict_ground_truth_4class,one_user_result_4class)
        head_line, value_line = compute_performance(gt4c,user4c,print_detail=0)
        if user_index==0:
            with open(csv_name,'a') as f:
                f.write('user_name'+','+head_line +'\n')
        with open(csv_name,'a') as f:
            f.write(str(user_index+1)+','+value_line+'\n')
    
    
    ## analysis result
    ## convert csv to excel
    excel_name = csv_name+'.xlsx'
    if os.path.exists(excel_name):
        os.remove(excel_name)
    df_new = pd.read_csv(csv_name)
    writer = pd.ExcelWriter(excel_name)
    df_new.to_excel(writer, index = False)
    writer.save()
    
    ## read excel to array
    result_df=pd.read_excel(excel_name)
    excel_head=list(result_df)
    result_array=result_df.to_numpy()
    print('mean acc')
    print(result_df['acc'].to_numpy().mean())
    print('mean total_auc')
    print(result_df['total_auc'].to_numpy().mean())
   
   
    ## high, 1-11, mid. 12-21, primary22-31
    ## high index, 0-10, mid. 11-20, primary21-30
    acc_HMP = calc_total_HMP(result_df,'acc')
    auc_HMP = calc_total_HMP(result_df,'total_auc')
    f1w_HMP = calc_total_HMP(result_df,'f1score_weigt')
    
    print('EXP = ',EXP)
    print('acc_HMP = \n',acc_HMP)
    print('auc_HMP = \n',auc_HMP)
    print('f1w_HMP = \n',f1w_HMP)
    print(np.hstack([acc_HMP,auc_HMP,f1w_HMP]))
    
    
    
    
    #############
    ## 2 Class ##
    #############
    ## [1] load ground truth make ground truth array, with name order
    ic_dict_ground_truth        = json.load(open(os.path.join(path_IC_Score_Pathologist ,"ic_dict_ground_truth.json"),'r'))
    ic_dict_ground_truth_2class = json.load(open(os.path.join(path_IC_Score_Pathologist ,"ic_dict_ground_truth_2class.json"),'r'))
    img_list_in_order           = np.load(os.path.join(path_IC_Score_Pathologist ,'img_list_in_order.npy')).tolist()
    
    
    ## [2] load algo, with pre-process 4 clsss, 2 class
    user31_result = json.load(open(os.path.join(path_IC_Score_Pathologist ,"ic_dict_medicine_exp"+str(EXP)+".json"),'r'))
    csv_name  = os.path.join(result_Calc,'evaluate_user31_exp'+str(EXP)+'_2class.csv')
    print("++"*100)
    print(csv_name)
    cutoff_2class=[0,1]
    
    
    if os.path.exists(csv_name):
        os.remove(csv_name)
    
    for user_index in range(0,31):
    #    print("_"*100)
    #    print('user_index = ',user_index)
        one_user_result_2class=dict()
        for one_key, one_value in user31_result.items():
            img_ic_score = one_value[user_index]
            one_user_result_2class.update({one_key: IC_one_value_to_group(img_ic_score,cutoff_2class)})
        
        
        ##  [3] #########   evaluation #####################
        ## convert dict 2 list with order
        gt2c,user2c =  ic_dict_to_array(ic_dict_ground_truth_2class,one_user_result_2class)
        head_line, value_line = compute_performance(gt2c,user2c,0)
        if user_index==0:
            with open(csv_name,'a') as f:
                f.write('user_name'+','+head_line +'\n')
        with open(csv_name,'a') as f:
            f.write(str(user_index+1)+','+value_line+'\n')
    
    ## analysis result
    ## convert csv to excel
    excel_name = csv_name+'.xlsx'
    if os.path.exists(excel_name):
        os.remove(excel_name)
    df_new = pd.read_csv(csv_name)
    writer = pd.ExcelWriter(excel_name)
    df_new.to_excel(writer, index = False)
    writer.save()
    
    ## read excel to array
    result_df=pd.read_excel(excel_name)
    excel_head=list(result_df)
    result_array=result_df.to_numpy()
    print('mean acc')
    print(result_df['acc'].to_numpy().mean())
    print('mean total_auc')
    print(result_df['total_auc'].to_numpy().mean())
    
    acc_HMP = calc_total_HMP(result_df,'acc')
    auc_HMP = calc_total_HMP(result_df,'total_auc')
    f1w_HMP = calc_total_HMP(result_df,'f1score_weigt')
    
    print('EXP = ',EXP)
    print('acc_HMP = \n',acc_HMP)
    print('auc_HMP = \n',auc_HMP)
    print('f1w_HMP = \n',f1w_HMP)
    print(np.hstack([acc_HMP,auc_HMP,f1w_HMP]))



###############################################################################
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y_%b_%d_%Hh%M")
print('Current Timestamp : ', timestampStr)

# calc for RS 1
calc_all(1)
# calc for RS 2
calc_all(2)
# calc for RS 3
calc_all(3)


