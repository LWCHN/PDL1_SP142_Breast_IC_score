import os
import pandas as pd
import numpy as np
from nipype.algorithms.icc import ICC_rep_anova
import json
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def resize_label_img(img,new_h,new_w):
    new_img = img.copy()
    new_img  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return new_img

def get_medicine_array(ic_dict, input_img_list,MEDICINE_NBR=31):
    img_list_raw=input_img_list.copy()
    ## make array for all other computes
    data_table_array=np.zeros([len(img_list_raw),MEDICINE_NBR])
    for i in range(0,len(img_list_raw)):
        for j in range(0,MEDICINE_NBR):
            user_name='test_medic_'+str(j+1)
            ###img_id = ic_dict[user_name][img_list_raw[i]]['ImgID']
            try:
                img_ic_score = ic_dict[user_name][img_list_raw[i]]['IcScore']
            except:
                print('________ERROR__________: def get_medicine_array(): user_name = ',user_name)
                print('________ERROR__________: def get_medicine_array(): img_list_raw[i]= ',img_list_raw[i])
                img_ic_score = None
                
            data_table_array[i,j]=float(img_ic_score)
    return data_table_array

def convert_score_dict_to_array(ic_dict, input_img_list,MEDICINE_NBR=31):
    img_list_raw=input_img_list.copy()
    ## make array for all other computes
    data_table_array=np.zeros([len(img_list_raw),MEDICINE_NBR])
    for i in range(0,len(img_list_raw)):
        for j in range(0,MEDICINE_NBR):
            try:
                img_ic_score = ic_dict[img_list_raw[i]][j]
            except:
                print('________ERROR__________: def get_medicine_array(): user_index = ',j)
                print('________ERROR__________: def get_medicine_array(): img_list_raw[i]= ',img_list_raw[i])
                img_ic_score = None
                
            data_table_array[i,j]=float(img_ic_score)
    return data_table_array

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

def IC_one_value_to_group(one_value,cutoff):
    ic_reslut_N_class=0
    for kk in range(0,len(cutoff)-1):
        if one_value>=cutoff[kk] and one_value<cutoff[kk+1]:
            ic_reslut_N_class=int(kk)
    if one_value>=cutoff[-1]:
        ic_reslut_N_class=int(len(cutoff)-1)
    return ic_reslut_N_class

def sort_list_by_array_mean(data_table_array,img_list_raw):
    ## make_order_of_data
    mean_array_all_user = data_table_array.mean(axis=1)
    list_mean_ic=list(mean_array_all_user)
    list_mean_ic.sort()
    img_list_order=list()
    for i in range(0,len(list_mean_ic)):
        one_mean_ic = list_mean_ic[i]
        index = np.where(mean_array_all_user==one_mean_ic)[0][0]
        #print('one_mean_ic,index = ',one_mean_ic,index)
        img_list_order.append(img_list_raw[index])
        mean_array_all_user[index]=None # mean_array_all_user contains same value, save index will be refind

    assert (  len(list(set(img_list_order))) == len(img_list_raw))
    return img_list_order
    
    
def raw_csv_2_json_dict(raw_csv_name, output_name):
    assert(os.path.exists(raw_csv_name))
    if 1:
        file = pd.read_csv(raw_csv_name)  
        df=pd.DataFrame(file)
        head_line=df[0:1]
        head_line.keys()
        
        ic_dict=dict()
        img_list=list()
        
        ## get user name , use for each element of dict
        for i in range(0,len(df)):
            line = df[i:i+1]
            ele_UserName = line['submittier'][i]
#            print('ele_UserName = ',ele_UserName)
            ele_UserName = ele_UserName.replace(" ", "")
            if ele_UserName[-1]==" ":
#                print('______________ele_UserName = ',ele_UserName)
                pdb.set_trace()
            ic_dict.update({ele_UserName:{}})
        print(ic_dict.keys())
        
        ## fill in ic value for each user name
        for i in range(0,len(df)):
            line = df[i:i+1]
            ele_ImgName = line['img_name'][i]
            ele_Reslut = line['result'][i]
            ele_ImgId = line['img_id'][i]
#            print(ele_ImgName)
            ele_UserName = line['submittier'][i]
            ele_UserName=ele_UserName.replace(" ", "") 
            one_item={ (os.path.basename(ele_ImgName)).split('.png')[0] : {'ImgID':int(ele_ImgId),'IcScore':float(ele_Reslut)}}
            ic_dict[ele_UserName].update(one_item)
            
            img_list.append(  (os.path.basename(ele_ImgName)).split('.png')[0]  )
        
        try:
            print( len(ic_dict['test_medic_1'].keys()))
        except:
            print('ele_UserName = ',ele_UserName)
            print( len(ic_dict[ele_UserName].keys()))

        with open(output_name,'w') as fp:
            json.dump(ic_dict,fp)
        
        return output_name, list(set(img_list))


def save_df_to_xlsx(df, save_name):
    try:
        writer = pd.ExcelWriter(save_name)
        df.to_excel(writer, 'page_1', float_format='%.6f')		# 'page_1' is sheet name
        writer.save()
        writer.close()
        return True
    except:
        print("______Error def save_df_to_xlsx()")
        return False

def json_to_array(img_list_path,json_data_path):
    img_list_in_order=(np.load(img_list_path)).tolist()
    ic_dict_medicine_exp1 = json.load(open(json_data_path,'r'))
    
    data_table_exp1=np.zeros([len(img_list_in_order),31+0])
    for i in range(0,len(img_list_in_order)):
        img_name=img_list_in_order[i]
        one_row_medicine=np.array(ic_dict_medicine_exp1[img_name])
        data_table_exp1[i,0:len(one_row_medicine)] = one_row_medicine[:]
    return data_table_exp1
    
def save_icc21_array_to_excel(name_excel,icc_bt_exp):
    print(name_excel, icc_bt_exp)
    data_table_head_list=list()
    MEDICINE_NBR = icc_bt_exp.shape[1]
    for i in range(0,MEDICINE_NBR):
        data_table_head_list.append('test_medic_'+str(i+1))
    data = pd.DataFrame(icc_bt_exp,index=['ICC(2,1)', 'CI95_LOW','CI95_UP'],columns=data_table_head_list)
    save_df_to_xlsx(data, name_excel)
    np.save(name_excel+'.npy',icc_bt_exp)
    
def ic_dict_to_array_with_img_name_root(dict_gt,dict_pred):
    gt_list=list()
    pred_list=list()
    for key_pred, value_pred in dict_pred.items():
        search_key = key_pred
        key_in_gt = [key for key, val in dict_gt.items() if search_key in key] 
        
        gt_list.append(dict_gt[key_in_gt[0]])
        pred_list.append(value_pred)
    return np.array(gt_list), np.array(pred_list)
