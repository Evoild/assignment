import torch
import torch.nn as nn
import torchvision
import sys
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def pil_to_np(img_PIL):
    '''
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.ndim == 2:
        ar = ar
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def images_np_load(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            print(f"已加载: {filename}, 尺寸: {img.size}")
            img = pil_to_np(img)
            images.append(img)
            print(img.shape)
    return images

def images_np_save(images, folder_path):
    idx = 0
    # 将图像保存为新文件
    for img in images:
        img = np_to_pil(img)
        img.save(folder_path + f"{idx}.jpg")
        idx += 1

def image_plot(image):
    ## modify the shape to H*W*C
    if image.shape[0] == 3:
        img = np.transpose(image, [1, 2, 0])
    else:
        img = image

    if img.ndim == 3:
        plt.imshow(img)
    elif img.ndim == 2:
        plt.imshow(img, cmap="gray")

    plt.axis('off')
    plt.show()

def images_plot(images):
    for image in images:
        image_plot(image)