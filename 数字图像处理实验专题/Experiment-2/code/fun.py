import numpy as np
from utils import *
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def get_region(c, size, image):

    if len(image.shape) == 3:
        return image[c[0] - size // 2: c[0] + size // 2, c[1] - size// 2: c[1] + size // 2, :]
    else:
        return image[c[0] - size // 2: c[0] + size // 2, c[1] - size// 2: c[1] + size // 2]


def patch_sim_get(image, x1, x2, patchsize, alpha=0.1):
    region1 = get_region(x1, patchsize, image)
    region2 = get_region(x2, patchsize, image)
    sim = np.exp(-alpha * np.sum((region1 - region2) ** 2))

    return sim

def self_loc_sim_get(image, regionsize=81, patchsize=5):
    H, W = image.shape[0], image.shape[1]
    region_r, patch_r = regionsize // 2, patchsize // 2

    similiarity_map = [[0 for _ in range(0, W - regionsize + 1)] for _ in range(0, H - regionsize + 1)]

    for i in range(0, H - regionsize + 1):
        for j in range(0, W - regionsize + 1):
            print(i, j)
            xc = [i + region_r, j + region_r]
            sim = np.zeros((regionsize - patchsize + 1, regionsize - patchsize + 1)).astype(np.float32)
            
            for m in range(regionsize - patchsize + 1):
                for n in range(regionsize - patchsize + 1):
                    y = [i + m + patch_r, j + n + patch_r]
                    sim[m][n] = patch_sim_get(image, xc, y, patchsize)
            similiarity_map[i][j] = sim

    return np.array(similiarity_map)


def log_polar_offsets(regionsize, num_r=4, num_theta=20):
    """生成 log-polar 相对坐标偏移表"""
    r = np.logspace(0, np.log10(regionsize / 2), num_r) - 1
    theta = np.linspace(0, 2*np.pi, num_theta, endpoint=False)

    rr, tt = np.meshgrid(r, theta, indexing='ij')
    dy = (rr * np.sin(tt)).astype(int)
    dx = (rr * np.cos(tt)).astype(int)
    offsets = np.stack([dy, dx], axis=-1)  

    return offsets


def log_polar_conversion(size, num_r=4, num_theta=20):

    R = size // 2
    r = np.linspace(1, R, num_r + 1)
    theta = np.linspace(-2*np.pi / (num_theta * 2), 2*np.pi - 2*np.pi / (num_theta * 2), num_theta + 1)
    
    return r[1:], theta[1:]


def update_log_polar_map(similarity_map, r, theta, num_r=4, num_theta=20):

    H, W = similarity_map.shape
    log_polar_map = np.zeros((num_r, num_theta)) 
    center_y, center_x = H // 2, W // 2  
    
    for i in range(H):
        for j in range(W):
            print(i,j)
            dy, dx = center_y - i, j - center_x
            r_val, t_val = np.sqrt(dy**2 + dx**2), (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
            if t_val > r[-1]:
                t_val = t_val - 2*np.pi
    
            r_idx, t_idx = np.digitize(r_val, r), np.digitize(t_val, theta) % num_theta  
            if r_idx < num_r:
                log_polar_map[r_idx, t_idx] = max(log_polar_map[r_idx, t_idx], similarity_map[i, j])
    
    return log_polar_map


def visualize_logpolar_feature(sim_feature, r):
    _ , num_theta = sim_feature.shape 
    theta = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    Theta, R = np.meshgrid(theta, r, indexing='xy')
    # 创建极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # 绘制扇形图
    c = ax.pcolormesh(Theta, R, sim_feature, shading='auto', cmap='gray')
    ax.set_theta_zero_location("E")  
    ax.set_theta_direction(1)        
    fig.colorbar(c, ax=ax, label='Similarity')
    plt.savefig('../result/sim_logpolar.png')
    plt.show()


def visualize_similarity_map(similarity_map):
    plt.imshow(similarity_map, cmap='gray')  
    plt.colorbar()  
    plt.title("Similarity Map") 
    plt.axis('off')  
    plt.savefig('../result/sim_ori.png')
    plt.show()


def self_loc_sim_get_polar(image, regionsize=81, patchsize=5, num_r=4, num_theta=20):
    
    H, W = image.shape[0], image.shape[1]
    h, w = H - regionsize + 1, W - regionsize + 1
    h, w = (H - regionsize) // patchsize + 1, (W - regionsize) // patchsize + 1

    region_r, patch_r = regionsize // 2, patchsize // 2
    r, theta = log_polar_conversion(size = regionsize - patchsize + 1)
    log_polar_maps = np.zeros((h, w, num_r, num_theta))

    for i in range(0, H - regionsize + 1, patchsize):
        for j in range(0, W - regionsize + 1, patchsize):
            xc = [i + region_r, j + region_r]
            
            for m in range(regionsize - patchsize + 1):
                for n in range(regionsize - patchsize + 1):

                    xh = [i + m + patch_r, j + n + patch_r]
                    dy, dx = xc[0] - xh[0], xh[1] - xc[1]

                    r_val, t_val = np.sqrt(dy**2 + dx**2), (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
                    if t_val > r[-1]: t_val = t_val - 2*np.pi
                    
                    r_idx, t_idx = np.digitize(r_val, r), np.digitize(t_val, theta) % num_theta  
                    
                    sim = patch_sim_get(image, xc, xh, patchsize)
                    if r_idx < num_r:
                        log_polar_maps[i // patchsize, j//patchsize, r_idx, t_idx] = max(log_polar_maps[i // patchsize, j//patchsize, r_idx, t_idx], sim)
            


    return log_polar_maps


if __name__ == '__main__':
    path = os.getcwd() + '/../data/'  
    images_np = images_np_load(path, plot=False)


    res = self_loc_sim_get_polar(np.transpose(images_np[0], [1, 2, 0]))
    visualize_similarity_map(self_loc_sim_get_polar(res[20][20]))
