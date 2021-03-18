# -*- coding: utf-8 -*-
"""
@author: leonlwang@tencent.com
"""
import os, sys
import imageio
from skimage.color import rgb2hsv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import square, disk

from matplotlib import rcParams
rcParams['font.family'] = "Times New Roman"

def image_read(img_path, mode="RGB"):
    mode = mode.lower()
    with open(img_path, 'rb') as fp:
        raw = fp.read()
        if mode == 'rgb':
            img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_COLOR)
            img = img[:,:,::-1]
        elif mode == 'gray' or mode == 'grey':
            img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_GRAYSCALE)
    return img

def _img2binary(rgb_img, 
                h_t=[101, 175], 
                s_t=[40, 120],
                v_t=[40, 150]):
    h, s, v = np.split(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV), 3, axis=-1)
    gray_img = (h > h_t[0])*(h < h_t[1]) * (s > s_t[0])*(s < s_t[1]) * (v > v_t[0])*(v < v_t[1])
    return gray_img.astype(np.float32)[..., 0],h[...,0],s[...,0],v[...,0]

def get_mask_morpho(binary_mask):
    stage1 = binary_mask
    stage1 = cv2.morphologyEx(stage1, cv2.MORPH_OPEN, square(1))
    
    kernel = square(2).astype(np.uint8)
    stage1 = cv2.dilate(stage1, kernel)
    kernel = disk(4).astype(np.uint8)
    Mask_morpho = cv2.dilate(stage1, kernel)
    return Mask_morpho

def overlay_mask(img, msk):
    lam = 0.5
    res = img.copy()
    msk = cv2.resize(msk, img.shape[:2][::-1])
    index = np.stack([msk > 0]*3, axis=-1)
    color = np.array([255, 0, 0])
    add_map = index * color.reshape((1, 1, 3))
    res[index] = img[index] * lam + (1 - lam) * add_map[index]
    return res

def cal_ICscore(b_img):
    return b_img.sum() * 1.0 / b_img.size


img_dir=r"./dataset_Fig2"
save_dir='./result_Fig'
if not os.path.exists(save_dir): os.mkdir(save_dir)
img_name=os.path.join(img_dir,"case_Fig2.png")
rawimg=image_read(img_name)
img = cv2.resize(rawimg, (1080, 1080))
mask_stain,h,s,v=_img2binary(img)
mask_morpho = get_mask_morpho(mask_stain)

mask_epithelium= imageio.imread(os.path.join(img_dir,"case_Fig2_epithelium_necrosis.png"))
mask_epithelium = cv2.resize(mask_epithelium, (1080, 1080), interpolation = cv2.INTER_NEAREST)

mask_IC = mask_morpho.copy()
mask_IC[np.where( mask_epithelium>0)] = 0
IC_value =  np.round(cal_ICscore(mask_IC),decimals=4)

## overlay result
overlay_mask_IC = overlay_mask(img, mask_IC)

save_nameroot = os.path.join(save_dir, '')
imageio.imwrite(save_nameroot+"Fig2a_Input_image.png", img)
imageio.imwrite(save_nameroot+"Fig2b_Hue_channel.png", h)
imageio.imwrite(save_nameroot+"Fig2c_Mask_stain.png", mask_stain)
imageio.imwrite(save_nameroot+"Fig2d_Mask_epithelium.png", mask_epithelium)
imageio.imwrite(save_nameroot+"Fig2e_Mask_IC--"+str(IC_value)+"--.png", mask_IC)
imageio.imwrite(save_nameroot+"Fig2f_overlay_Mask_IC--"+str(IC_value)+"--.png", overlay_mask_IC)



