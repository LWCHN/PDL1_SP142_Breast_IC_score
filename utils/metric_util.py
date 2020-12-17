import numpy as np
import cv2
from numpy.core.umath_tests import inner1d
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

def Dice(gt_mask,pd_mask):

    A, B = gt_mask>0, pd_mask>0
    insection = np.logical_and(A,B)
    overall_dice = 2* float(np.sum(insection))/(float(np.sum(A)) + float(np.sum(B)))

    return overall_dice

def Jaccard(gt_mask,pd_mask): #IOU

    A, B = gt_mask>0, pd_mask>0
    insection = np.logical_and(A,B)
    overall_jaccard = float(np.sum(insection))/float(np.sum(np.logical_or(A,B)))

    return overall_jaccard

def Object_Dice_and_Jaccard(gt_cnt,pd_cnt,gt_mask):

    gt_x, gt_y, gt_w, gt_h = cv2.boundingRect(gt_cnt)
    pd_x, pd_y, pd_w, pd_h = cv2.boundingRect(pd_cnt)

    if gt_x + gt_w < pd_x or pd_x + pd_w < gt_x or gt_y + gt_h < pd_y or pd_y + pd_h < gt_y:
        dice = 0
        jaccard = 0
        return dice, jaccard

    blank = np.zeros( gt_mask.shape[0:2], dtype = np.uint8 )
    img_gt = cv2.drawContours( blank.copy(), [gt_cnt], 0, 255, -1 )
    img_pd = cv2.drawContours( blank.copy(), [pd_cnt], 0, 255, -1 )

    dice = Dice(img_gt,img_pd)
    jaccard = Jaccard(img_gt,img_pd)
    return dice, jaccard



def Hausdorff(A,B):
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B) - 2*(np.dot(A,B.T)))
    dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
    return dH

def Hausdorff_Dist(gt_cnt,pd_cnt):

#    pd_x, pd_y = pd_cnt.T
#    gt_x, gt_y = gt_cnt.T
#    set_pd = zip(pd_x.tolist()[0],pd_y.tolist()[0])
#    set_gt = zip(gt_x.tolist()[0],gt_y.tolist()[0])
#
#    Hd = Hausdorff(np.array(set_pd), np.array(set_gt))
#    return Hd
    return None



def TPR_and_FPI(gt_mask, pd_mask, morph_size=3, THRESH=0.2):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_size,morph_size))

    gt_mask = cv2.morphologyEx(gt_mask.astype(np.float32), cv2.MORPH_OPEN, kernel)
    pd_mask = cv2.morphologyEx(pd_mask.astype(np.float32), cv2.MORPH_OPEN, kernel)
    gt_mask[gt_mask > 0] = 255
    pd_mask[pd_mask > 0] = 255

    _,gt_cnts,_ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _,pd_cnts,_ = cv2.findContours(pd_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(gt_cnts) == 0:
        TP = 0
        gt_number = 0
        FPI = len(pd_cnts)
        TPR = 0
        mean_object_dice = 0
        mean_object_jaccard = 0
        mean_hd = 0
        return TPR, FPI, TP, gt_number, mean_object_dice, mean_object_jaccard, mean_hd,None,None

    if len(pd_cnts) == 0:
        TP = 0
        gt_number = len(gt_cnts)
        FPI = 0
        TPR = 0
        mean_object_dice = 0
        mean_object_jaccard = 0
        mean_hd = 0
        return TPR, FPI, TP, gt_number, mean_object_dice, mean_object_jaccard, mean_hd,None,None

    FPI = 0
    gt_number = len(gt_cnts)
    tp_list = np.zeros(gt_number)
    dice_list = np.zeros(gt_number)
    jaccard_list = np.zeros(gt_number)
    hd_list = np.zeros(gt_number)

    for pd_cnt in pd_cnts:
        fp_flag = True
        for i, gt_cnt in enumerate(gt_cnts):

            dice, jaccard = Object_Dice_and_Jaccard(gt_cnt, pd_cnt,gt_mask)

            if dice > dice_list[i]:
                dice_list[i] = dice
                Hd = Hausdorff_Dist(gt_cnt,pd_cnt)
                hd_list[i] = Hd

            if jaccard > jaccard_list[i]:
                jaccard_list[i] = jaccard

            if jaccard > THRESH:
                tp_list[i] += 1
                fp_flag = False

        if fp_flag:
            FPI += 1

    TP = np.sum(tp_list!=0)
    TPR = float(TP) / float(gt_number)
    mean_object_dice = np.mean(dice_list)
    mean_object_jaccard = np.mean(jaccard_list)
    mean_hd=None
    #mean_hd = float(np.sum(hd_list)) / float(np.sum(hd_list!=0))

    return TPR, FPI, TP, gt_number, mean_object_dice, mean_object_jaccard, mean_hd,gt_mask,pd_mask


def __metrics_confu_mat(confu_mat_total, save_path=None):
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.00000001
    col_sum = np.sum(confu_mat, axis=1)  
    raw_sum = np.sum(confu_mat, axis=0) 
 
    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()
 
    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)
 

    TP = []
    for i in range(class_num):
        TP.append(confu_mat[i, i])
 
    # compute  f1-score
    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP
    
    # compute precision，recall, f1-score，f1-m, and mIOU
    f1_m = []
    iou_m = []
    for i in range(class_num):
        #f1-score
        f1 = 1.0*TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = 1.0*TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)
 
    f1_m = np.array(f1_m)
    iou_m = np.array(iou_m)
    
    
    mf1_score = np.mean(f1_m)
    mIou = np.mean(iou_m)
    
    precision = list()
    recall    = list()
    f1score = list()
    Iou=list()
    for i in range(class_num):
        precision.append(  float(TP[i])/raw_sum[i]  )
        recall.append(  float(TP[i])/col_sum[i]  )
        f1score.append(  float(f1_m[i])  )
        Iou.append(float(iou_m[i]) )
    
    
    
    result_list=[oa,kappa,mf1_score,mIou,precision,recall,f1score,Iou]
    if save_path is not None:
        with open(save_path + 'Output_def_metrics_confu_mat.txt', 'w') as f:
            f.write('OA(precision_micro):\t%.4f\n' % (oa*100))
            f.write('kappa:\t%.4f\n' % (kappa*100))
            f.write('mf1-score:\t%.4f\n' % (np.mean(f1_m)*100))
            f.write('mIou:\t%.4f\n' % (np.mean(iou_m)*100))
 
            #precision
            f.write('precision:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i]/raw_sum[i])*100))
            f.write('\n')
 
            #recall
            f.write('recall:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / col_sum[i])*100))
            f.write('\n')
 
            #f1-score
            f.write('f1-score:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(f1_m[i])*100))
            f.write('\n')
 
            #IOU
            f.write('Iou:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(iou_m[i])*100))
            f.write('\n')
    print('result_list = oa,kappa,mf1_score,mIou,precision,recall,f1score,Iou')
    reslut_name = ['oa','kappa','mf1_score','mIou','precision','recall','f1score','Iou']
    for i in range(0,len(result_list)):
        print(reslut_name[i])
        print( np.round(np.array(result_list[i]),decimals=3)  )
    return TP,FN,FP,result_list,reslut_name

    
def metrics_confu_mat_multi_class(cm):
    very_small_value=0.000000001
    TP = 1.0*np.diag(cm)
    FP = 1.0*cm.sum(axis=0) - np.diag(cm)  
    FN = 1.0*cm.sum(axis=1) - np.diag(cm)
    TN = 1.0*cm.sum() - (FP + FN + TP)
    
    # sum of the correct prediction for all class
    ACC_ALL=TP.sum()/cm.sum()
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN+very_small_value)
    # Specificity or true negative rate
    TNR = TN/(TN+FP+very_small_value) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP+very_small_value)
    # Negative predictive value
    NPV = TN/(TN+FN+very_small_value)
    # Fall out or false positive rate
    FPR = FP/(FP+TN+very_small_value)
    # False negative rate
    FNR = FN/(TP+FN+very_small_value)
    # False discovery rate
    FDR = FP/(TP+FP+very_small_value)
    
    F1score= (2.0*TPR * PPV) / (TPR+PPV+very_small_value)
    #precision = TP / (TP+FP)  #
    #recall = TP / (TP+FN)  #
    
    result_list=[TP,FP,FN,TN,ACC_ALL,TPR,TNR,PPV,NPV,FPR,FNR,FDR,F1score]
    ###result_name=['TP','FP','FN','TN','ACC_ALL','TPR(Recall,Sensitivity)','TNR(Specificity)','PPV(Precision)','NPV','FPR','FNR','FDR','F1score']
    result_name_dict = {'TP':TP,'FP':FP,'FN':FN,'TN':TN,'ACC_ALL':ACC_ALL,'TPR_Recall_Sensitivity':TPR,'TNR_Specificity':TNR,'PPV_Precision':PPV,'NPV':NPV,'FPR':FPR,'FNR':FNR,'FDR':FDR,'F1score':F1score}
    return result_list,result_name_dict
    
def multiclass_roc_auc_score(truth, pred, decimals=4,average="macro"):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    total_score = roc_auc_score(truth, pred, average=average)
    details_score = list()
    for i in range(0,truth.shape[1]):
        details_score.append(roc_auc_score(truth[:,i], pred[:,i], average=average))
    details_score = np.array(details_score)
    return np.round(total_score,decimals=decimals), np.round(details_score,decimals=decimals)

def __roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):
#creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]
    
        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
    
        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    return roc_auc_dict
    
def get_evaluation_binary(ic_result, y_true):
    
    try:
        array_eva=np.zeros([7,ic_result.shape[1]])
        all_usr_nbr=ic_result.shape[1]
    except:
        array_eva=np.zeros([7,1])
        all_usr_nbr=1
    for user_index in range(0,all_usr_nbr):
        print('_'*30)
        print('user_index',user_index)
        try:
            y_pred = ic_result[:,user_index ]
        except:
            y_pred = ic_result
        TN,FP,FN,TP = metrics.confusion_matrix(y_true, y_pred).ravel()
        print('TN,FP,FN,TP = ',TN,FP,FN,TP)
        
        Accuracy=1.0*(TP+TN)/(TP+FP+FN+TN)
        Recall = 1.0*(TP)/(TP+FN++0.00001)
        Precision = 1.0*(TP)/(TP+FP)
        FP= FP
        FalsePositiveRate =FP/(FP+TN+0.00001)*1.0
        Sensitivity =1.0* TP/(TP+FN+0.00001)
        Specificity = 1-FP/(FP+TN+0.00001)*1.0
        
        array_eva[:,user_index]=np.round(np.array([Accuracy,Recall,Precision,FP,FalsePositiveRate,Sensitivity,Specificity]),decimals=3)
        array_eva_name = ['Accuracy','Recall','Precision','FP','FalsePositiveRate','Sensitivity','Specificity']
    return array_eva,array_eva_name