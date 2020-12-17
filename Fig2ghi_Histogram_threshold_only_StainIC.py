# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:29:23 2020
@author: leonlwang
"""
import os, sys,pdb
import imageio
import pandas as pd
from skimage.color import rgb2hsv
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
##https://www.icaml.org/canon/data/images-videos/HSV_color_space/HSV_color_space.html
from skimage.morphology import square, disk


from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman" 
from utils import path_util
def _img2binary(rgb_img, 
                h_t=[101, 175], 
                s_t=[40, 120],
                v_t=[40, 150]):
    h, s, v = np.split(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV), 3, axis=-1)
    gray_img = (h > h_t[0])*(h < h_t[1]) * (s > s_t[0])*(s < s_t[1]) * (v > v_t[0])*(v < v_t[1])
    return gray_img.astype(np.float32)[..., 0],h[...,0],s[...,0],v[...,0]

def calc_hist_curve(list2D):
    array2D=np.asarray(list2D)
    return  array2D.sum(axis=0)/array2D.shape[0]

def get_hist_data(channel_img,channel_list):
    hist_data = cv2.calcHist([channel_img],
                          [0],
                          None,
                          [hist_bin],
                          [int(0),int(hist_bin-1)])[...,0]
    channel_list = channel_list+np.array(hist_data).reshape((len(hist_data),1))
    return channel_list

def get_mask_morpho(binary_mask):
    stage1 = binary_mask
    stage1 = cv2.morphologyEx(stage1, cv2.MORPH_OPEN, square(1))
    
    kernel = square(2).astype(np.uint8)
    stage1 = cv2.dilate(stage1, kernel)
    kernel = disk(4).astype(np.uint8)
    Mask_morpho = cv2.dilate(stage1, kernel)
    return Mask_morpho

## roche 25 images
rawimg_dir = os.path.join(r'./dataset/Roche_25img/raw_img','')
epithelium_dir = os.path.join(r'./dataset/Roche_25img/epithelium_mask','')
# gt = pd.read_excel(os.path.join('./dataset/Roche_25img/GroundTruth','dataset_Roche_25img_groundtruth.xlsx'))


save_path='./result_Fig'
path_util.makeDIR(save_path)
img_path_list,_= path_util.dir_reader(rawimg_dir, "(.*).png")
h_t=[101, 175]
s_t=[40, 120]
v_t=[40, 150]



hist_bin=256
h_hist=np.zeros([hist_bin,1])
s_hist=np.zeros([hist_bin,1])
v_hist=np.zeros([hist_bin,1])
score_list=list()
for i in range(0,len(img_path_list)):
    print('*'*100,'\n','i=',i)
    img_name=(os.path.basename(img_path_list[i])).split('.png')[0]
    rawimg = imageio.imread(img_path_list[i])
    img = cv2.resize(rawimg, (1080, 1080))
    mask_stain,h,s,v=_img2binary(img,h_t,s_t,v_t)
    
    img_name_epithelium_list,_ = path_util.dir_reader(epithelium_dir, '(.*)'+img_name+'(.*)'+'.png')
    img_name_epithelium = img_name_epithelium_list[0]
    os.path.exists(img_name_epithelium)
    mask_ephithemlium= imageio.imread(img_name_epithelium)
    mask_ephithemlium = cv2.resize(mask_ephithemlium, (1080, 1080), interpolation = cv2.INTER_NEAREST)
    
    mask_stain_filtered = mask_stain.copy()
    mask_stain_filtered[np.where( mask_ephithemlium>0)] = 0
    mask_IC_morpho = mask_stain_filtered.copy()

    h=h*mask_IC_morpho
    s=s*mask_IC_morpho
    v=v*mask_IC_morpho
    
    h_hist = get_hist_data(h,h_hist)
    s_hist = get_hist_data(s,s_hist)
    v_hist = get_hist_data(v,v_hist)
    
## save into numpy
np.savez(os.path.join(save_path,'hsv_hist.npz'),h_hist=h_hist,s_hist=s_hist,v_hist=v_hist)
data=np.load(os.path.join(save_path,'hsv_hist.npz'))


plt.close('all')
FONTSIZE=22
fig_PAPER=plt.figure(num='hue')
fig_PAPER.set_size_inches(6, 4)
plt.plot( h_hist,  'k-')
plt.axvline(h_t[0],color='gray',linestyle='--')
plt.axvline(h_t[1],color='gray',linestyle='--')
plt.title('Histogram of hue channel',fontsize=FONTSIZE)
ax=plt.gca();
plt.xlim([1,256])
plt.ylim([0,h_hist[1:].max()])
ax.set_xlabel('Pixel value', labelpad=-3,fontsize=FONTSIZE)
ax.set_ylabel('Pixel number', labelpad=-4,fontsize=FONTSIZE)
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
fig_PAPER.tight_layout()
fig_PAPER.savefig(os.path.join(save_path,"Fig2g_algo_histogram_Hue.png"),dpi=300)
fig_PAPER.savefig(os.path.join(save_path,"Fig2g_algo_histogram_Hue.pdf"),dpi=300)

plt.close('all')
fig_PAPER=plt.figure(num='satuation')
fig_PAPER.set_size_inches(6, 4)
plt.plot( s_hist,  'k-')
plt.axvline(s_t[0],color='gray',linestyle='--')
plt.axvline(s_t[1],color='gray',linestyle='--')
plt.title('Histogram of satuation channel',fontsize=FONTSIZE)
ax=plt.gca();
plt.xlim([1,256])
plt.ylim([0,s_hist[1:].max()])
ax.set_xlabel('Pixel value', labelpad=-3,fontsize=FONTSIZE)
ax.set_ylabel('Pixel number', labelpad=-4,fontsize=FONTSIZE)
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
fig_PAPER.tight_layout()
fig_PAPER.savefig(os.path.join(save_path,"Fig2h_algo_histogram_Satuation.png"),dpi=300)
fig_PAPER.savefig(os.path.join(save_path,"Fig2h_algo_histogram_Satuation.pdf"),dpi=300)


plt.close('all')
fig_PAPER=plt.figure(num='value')
fig_PAPER.set_size_inches(6, 4)
plt.plot(v_hist,  'k-')
plt.axvline(v_t[0],color='gray',linestyle='--')
plt.axvline(v_t[1],color='gray',linestyle='--')
plt.title('Histogram of value channel',fontsize=FONTSIZE)
ax=plt.gca();
plt.xlim([1,256])
plt.ylim([0,v_hist[1:].max()])
ax.set_xlabel('Pixel value', labelpad=-3,fontsize=FONTSIZE)
ax.set_ylabel('Pixel number', labelpad=-4,fontsize=FONTSIZE)
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
fig_PAPER.tight_layout()
fig_PAPER.savefig(os.path.join(save_path,"Fig2i_algo_histogram_Value.png"),dpi=300)
fig_PAPER.savefig(os.path.join(save_path,"Fig2i_algo_histogram_Value.pdf"),dpi=300)
