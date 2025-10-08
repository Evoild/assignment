#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision
from utils import *
from scipy.sparse import spdiags, diags, csr_matrix, csc_matrix
import cv2


##1. 求暗通道
def dark_channel(image, radius=15):
    min_rgb, _ = torch.min(image, dim=1, keepdim=True)  # shape: (B, 1, H, W)
    kernel_size = 2 * radius + 1  
    padding = radius  
    dark_channel = -F.max_pool2d(-min_rgb, 
                                 kernel_size=kernel_size, 
                                 stride=1, 
                                 padding=padding)
#    dark_channel = -F.max_pool2d(-image, 
#                                 kernel_size=kernel_size, 
#                                 stride=1, 
#                                 padding=padding)
#    dark_channel, _ = torch.min(dark_channel, dim=1, keepdim=True)
    
    return dark_channel


##2. 估计大气亮度
def estimate_atmospheric_light(dark_channel, haze_img):
    H, W = dark_channel.shape
    if haze_img.shape != (3, H, W):
        raise ValueError(f"haze_img形状应为 (3, {H}, {W})，但得到 {haze_img.shape}")
    
    threshold = np.percentile(dark_channel, 99.9)
    mask = dark_channel >= threshold
    # 获取每个通道的掩码区域, 在三个通道的所有前0.1%像素中找到最大值
    top_pixels_rgb = [haze_img[i, mask] for i in range(3)]     
    atmospheric_light = [np.max(top_pixels_rgb[i]) for i in range(3)]
    
    return atmospheric_light

##3、计算传导图像
def compute_transmission_simple(dark_channel, atmospheric_light, omega=0.95):
    if not np.isscalar(atmospheric_light):
        A = np.max(atmospheric_light)
    else:
        A = atmospheric_light
    # 计算传导图像
    transmission = 1 - omega * (dark_channel / A)
    transmission = np.clip(transmission, 0.1, 1.0)
    
    return transmission


##4. 恢复图像
def recover_scene_image(haze_img, transmission, atmospheric_light, t0=0.1):
    C, H, W = haze_img.shape
    
    if np.isscalar(atmospheric_light):
        A = np.array([atmospheric_light, atmospheric_light, atmospheric_light])
    else:
        A = np.array(atmospheric_light)
        if len(A) != 3:
            raise ValueError(f"atmospheric_light应为长度为3的数组，但得到 {len(A)}")
    
    A = A.reshape(3, 1, 1)
    
    if transmission.shape != (H, W):
        raise ValueError(f"transmission形状应为 ({H}, {W})，但得到 {transmission.shape}")
    
    # 对传导图像应用下界阈值
    t_clipped = np.maximum(transmission, t0)
    # 恢复原始图像: J = (I - A) / t + A
    scene_img = (haze_img - A) / t_clipped + A
    scene_img = np.clip(scene_img, 0, 255)
    
    return scene_img


def recover_scene_image_uint8(haze_img, transmission, atmospheric_light, t0=0.1):

    scene_img = recover_scene_image(haze_img, transmission, atmospheric_light, t0)
    return scene_img.astype(np.uint8)


##5. 优化传导图像
def laplacian_filter(image):
    """应用拉普拉斯算子进行滤波"""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

def soft_matting_with_laplacian(image, transmission, max_iter=30, epsilon=1e-3):

    height, width = image.shape[:2]
    # 初始透射率估计
    transmission = transmission / np.max(transmission)
    
    # 反向传播迭代
    for _ in range(max_iter):
        # 计算透射率的拉普拉斯算子
        laplacian_transmission = laplacian_filter(transmission)
        
        # 根据拉普拉斯算子调整透射率
        transmission_new = transmission - epsilon * laplacian_transmission
        
        # 保证透射率在合理范围内
        transmission_new = np.clip(transmission_new, 0, 1)
        
        # 如果变化小于阈值，停止迭代
        if np.max(np.abs(transmission_new - transmission)) < epsilon:
            break
        transmission = transmission_new

    return transmission

if __name__ == "__main__":
    #step0: load the images with form of np.array, set the prams
    images_np = images_np_load(os.getcwd())
    #images_plot(images_np)
    radius_min_filter = 3
    save_path_dark_channel = 'darkchannel/'
    save_path_transmission = 'transmission/'
    save_path_recover = 'recover/'
    save_path_softmatting = 'softmatting/'
    save_path_recover_soft = 'recover2/'


    ## step1: caculate the Dark Channel Prior 
    images_torch = [np_to_torch(img) for img in images_np]

    images_dark_torch = [dark_channel(image=img, radius=radius_min_filter) for img in images_torch]
    images_dark_np = [np.squeeze(torch_to_np(img)) for img in images_dark_torch]
    #images_plot(images_dark_np)
    images_np_save(images_dark_np, save_path_dark_channel)


    ## step2: caculate the atmospheric light
    atmo_light = [estimate_atmospheric_light(img_dark, img_haze) for img_dark, img_haze in zip(images_dark_np, images_np)] 


    ## step3: Estimating the Transmission
    tran_map = [compute_transmission_simple(image_dark, alight) for image_dark, alight in zip(images_dark_np, atmo_light)]
    images_np_save(tran_map, save_path_transmission)
    #images_plot(tran_map)


    ## step4: recover the Scene
    images_recover_np = [recover_scene_image(haze_img, transmission, atmospheric_light) for haze_img, transmission, atmospheric_light in zip(images_np, tran_map, atmo_light)]
    images_np_save(images_recover_np, save_path_recover)
    #images_plot(images_recover_np)


    ## step5: recover the Scene with Soft Matting 
    tran_map_opt = [soft_matting_with_laplacian(I, t) for t, I in zip(tran_map, images_np)]
    images_np_save(tran_map_opt, save_path_softmatting)
    images_recover_opt_np = [recover_scene_image(haze_img, transmission, atmospheric_light) for haze_img, transmission, atmospheric_light in zip(images_np, tran_map_opt, atmo_light)]
    images_np_save(images_recover_opt_np, save_path_recover_soft)
    #images_plot(images_recover_opt_np)

