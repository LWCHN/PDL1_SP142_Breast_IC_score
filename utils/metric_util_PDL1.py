import sys,os,pdb
from . import path_util, metric_util
import cv2
import numpy as np
import imageio
from skimage.morphology import square, disk
import pandas as pd
import json
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from skimage import img_as_ubyte



def IC_class_to_group(data_table_array,cutoff):
    ic_reslut_N_class=np.zeros_like(data_table_array)
    ic_reslut_N_class.astype(np.uint8)
    for i in range(0,data_table_array.shape[0]):
        for j in range(0,data_table_array.shape[1]):
            for kk in range(0,len(cutoff)-1):
                if data_table_array[i,j]>=cutoff[kk] and data_table_array[i,j]<cutoff[kk+1]:
                    ic_reslut_N_class[i,j]=int(kk)
            if data_table_array[i,j]>=cutoff[-1]:
                ic_reslut_N_class[i,j]=int(len(cutoff)-1)
    return ic_reslut_N_class


def compute_performance(array_gt, array_pred, print_detail=0):
    #(1)
    CM_all_class = metrics.confusion_matrix(array_gt,array_pred)
    value,name_dict = metric_util.metrics_confu_mat_multi_class(CM_all_class)
    name=list(name_dict.keys())
    if print_detail:
        print('\n +++++++++++++ metric_util.metrics_confu_mat_multi_class +++++++++++++\n')
        for i in range(0,len(name)):
            print('                  ',name[i],'=',np.round(value[i],decimals=3))
        print('\n +++++++++++++ ++++++++++++++++++++++++++++++++++++++++\n')
        print('\n CM_all_class = \n',CM_all_class)
    #(2)
    sk_report_all_class = sklearn.metrics.classification_report(digits=3,y_true=array_gt,y_pred=array_pred)
    if print_detail: print('\n sk_report_all_class = \n',sk_report_all_class)
    #(3)
    total_auc,detail_auc = metric_util.multiclass_roc_auc_score(array_gt,  array_pred,decimals=3)
    if print_detail: print('\n total_auc,detail_auc = \n',total_auc,detail_auc)
    #(4)
    kappa_all_class = sklearn.metrics.cohen_kappa_score(array_gt,  array_pred)
    if print_detail: print('\n kappa_all_class = ',np.round(kappa_all_class,decimals=3))
    if print_detail: print('0.61---0.80: Substantial agreement')
    if print_detail: print('0.81---0.99: Almost perfect agreement')
    

    ## (5) generate formated result
    prec_weigt  = sklearn.metrics.precision_score(array_gt,  array_pred, average='weighted')
    prec_macro  = sklearn.metrics.precision_score(array_gt,  array_pred, average='macro')
    
    acc         = name_dict['ACC_ALL']
    total_auc   = total_auc
    kappa       = kappa_all_class
    prec_macro    = name_dict['PPV_Precision'].mean()
    prec_weigt    = (name_dict['PPV_Precision'] * CM_all_class.sum(axis=1) ).sum() /CM_all_class.sum()
    recall_macro  = name_dict['TPR_Recall_Sensitivity'].mean()
    recall_weigt  = (name_dict['TPR_Recall_Sensitivity'] * CM_all_class.sum(axis=1) ).sum() /CM_all_class.sum()
    f1score_macro = name_dict['F1score'].mean()
    f1score_weigt = (name_dict['F1score'] * CM_all_class.sum(axis=1) ).sum() /CM_all_class.sum()
    
    if print_detail: print('acc,total_auc,kappa,prec_macro,prec_weigt,recall_macro,recall_weigt,f1score_macro,f1score_weigt')
    if print_detail: print(np.round(np.array([acc,total_auc,kappa,prec_macro,prec_weigt,recall_macro,recall_weigt,f1score_macro,f1score_weigt]),decimals=3))
    line_head = 'acc'+','+'total_auc'+','+'kappa'+','+'prec_macro'+','+'prec_weigt'+','+'recall_macro'+','+'recall_weigt'+','+'f1score_macro'+','+'f1score_weigt'
    line_value = str(acc)+','+str(total_auc)+','+str(kappa)+','+str(prec_macro)+','+str(prec_weigt)+','+str(recall_macro)+','+str(recall_weigt)+','+str(f1score_macro)+','+str(f1score_weigt)
    return line_head,line_value
    
    
    
def csv_to_excel(csv_name):
    ## convert csv to excel
    excel_name = csv_name+'.xlsx'
    if os.path.exists(excel_name):
        os.remove(excel_name)
    df_new = pd.read_csv(csv_name)
    writer = pd.ExcelWriter(excel_name)
    df_new.to_excel(writer, index = False)
    writer.save()
    return excel_name



def get_accuracy(gt_reslut,ic_reslut,cutoff_N_class,save_dir,save_name):
    gt_reslut_4c = IC_class_to_group(gt_reslut ,cutoff_N_class)
    ic_reslut_4c = IC_class_to_group(ic_reslut ,cutoff_N_class)
    head_line, value_line = compute_performance(gt_reslut_4c,ic_reslut_4c)
    csv_name = os.path.join(save_dir,save_name)
    if os.path.exists(csv_name):
        os.remove(csv_name)
    with open(csv_name ,'a+') as f:
        f.write(head_line +'\n')
        f.write(value_line+'\n')
    #excel_name = csv_to_excel(csv_name)
    #print('Accuracy save into: ',excel_name)


def evaluation_regression(pred_data,gt_data):
    ## https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    ## https://www.cnblogs.com/zzzzy/p/8490662.html
    
    #[1] R_Square
    import statsmodels.api as sm
    X_addC = sm.add_constant(pred_data)
    result = sm.OLS(gt_data, X_addC).fit()
    R_Square = result.rsquared
    
    #[2] MSE
    from sklearn.metrics import mean_squared_error
    import math
    
    MSE = mean_squared_error(gt_data, pred_data)
    RMSE = math.sqrt(mean_squared_error(gt_data, pred_data))
    
    #[3] MAE
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(gt_data, pred_data)
    
    return R_Square,MSE,RMSE,MAE


def  get_error_accuracy(gt_reslut,ic_reslut,save_dir,save_name):
    error_accuracy = evaluation_regression(ic_reslut/100.0,gt_reslut/100.0)
    error_accuracy = np.round(np.asarray(error_accuracy),decimals=6)
    print('R_Square,MSE,RMSE,MAE = ',error_accuracy) 
    head_line = 'R_Square,MSE,RMSE,MAE'
    value_line = ''
    for i in range(0, len(error_accuracy)):
        value_line=value_line+str(error_accuracy[i])+','
    csv_name = os.path.join(save_dir,save_name)
    with open(csv_name ,'w+') as f:
        f.write(head_line +'\n')
        f.write(value_line+'\n')
    #excel_name = csv_to_excel(csv_name)
    #print('get_error_accuracy: ',excel_name)
    
    