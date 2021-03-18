# -*- coding: utf-8 -*-
"""
@author: leonlwang@tencent.com
"""
import os,sys
import numpy as np
import json
from datetime import datetime
from utils.tools_pdl1_exp import IC_one_value_to_group,ic_dict_to_array_with_img_name_root
from utils.metric_util_PDL1 import compute_performance


###############################################################################
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y_%b_%d_%Hh%M")
print('Current Timestamp : ', timestampStr)

## [1] load ground truth make ground truth array, with name order
ic_dict_ground_truth        = json.load(open("./IC_Score_Pathologist/ic_dict_ground_truth.json",'r'))
ic_dict_ground_truth_4class = json.load(open("./IC_Score_Pathologist/ic_dict_ground_truth_4class.json",'r'))
ic_dict_ground_truth_2class = json.load(open("./IC_Score_Pathologist/ic_dict_ground_truth_2class.json",'r'))
img_list_in_order=np.load('./IC_Score_Pathologist/img_list_in_order.npy').tolist()


## [2] load algo, with pre-process 4 clsss, 2 class
algo_result = json.load(open('./Result_AI/IC_SCORE_AI_ImgID.csv.json','r'))

cutoff_4class=[0,1,5,10]
algo_result_4class=dict()
for one_key, one_value in algo_result.items():
    img_ic_score = one_value
    algo_result_4class.update({one_key: IC_one_value_to_group(img_ic_score,cutoff_4class)})
with open("./Result_AI/IC_SCORE_AI_4class.json",'w') as fp:
    json.dump(algo_result_4class,fp)

cutoff_2class=[0,1]
algo_result_2class=dict()
for one_key, one_value in algo_result.items():
    img_ic_score = one_value
    algo_result_2class.update({one_key: IC_one_value_to_group(img_ic_score,cutoff_2class)})
with open("./Result_AI/IC_SCORE_AI_2class.json",'w') as fp:
    json.dump(algo_result_2class,fp)


###############################################################################
##  [3] #########   evaluation #####################
gt4c,pred4c =  ic_dict_to_array_with_img_name_root(ic_dict_ground_truth_4class,algo_result_4class)
head_line, value_line = compute_performance(gt4c,pred4c)
csv_name = './Result_AI/evaluate_Algo_Ground_Truth_4class.csv'
if os.path.exists(csv_name):
    os.remove(csv_name)
with open(csv_name ,'a+') as f:
    f.write(head_line +'\n')
    f.write(value_line+'\n')


gt2c,pred2c =  ic_dict_to_array_with_img_name_root(ic_dict_ground_truth_2class,algo_result_2class)
head_line, value_line = compute_performance(gt2c,pred2c)
csv_name = './Result_AI/evaluate_Algo_Ground_Truth_2class.csv'
if os.path.exists(csv_name):
    os.remove(csv_name)
with open(csv_name ,'a+') as f:
    f.write(head_line +'\n')
    f.write(value_line+'\n')





