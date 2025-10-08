import numpy as np
from scipy.sparse import spdiags, linalg
from scipy.sparse import lil_matrix
import cv2

def get_laplacian(I, epsilon=1e-7, win_size=1):
    """
    计算拉普拉斯矩阵 A
    
    :param I: 输入图像，shape 为 (H, W, C)
    :param epsilon: 平滑因子，避免方差为零
    :param win_size: 邻域窗口大小
    :return: 稀疏矩阵 A
    """
    h, w, c = I.shape
    img_size = h * w
    neb_size = (win_size * 2 + 1) ** 2

    # 用来存储稀疏矩阵的数据
    row_inds = []
    col_inds = []
    vals = []

    # 获取图像索引矩阵
    indsM = np.reshape(np.arange(img_size), (h, w))
    
    # 构建拉普拉斯矩阵
    for j in range(win_size, w - win_size):
        for i in range(win_size, h - win_size):
            # 获取当前窗口的索引
            win_inds = indsM[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1].flatten()
            
            # 获取当前窗口的图像区域
            winI = I[i - win_size:i + win_size + 1, j - win_size:j + win_size + 1, :]
            winI = winI.reshape(neb_size, c)
            
            # 计算窗口内的均值和方差
            win_mu = np.mean(winI, axis=0)
            win_var = (np.dot(winI.T, winI) / neb_size) - np.outer(win_mu, win_mu) + (epsilon / neb_size) * np.eye(c)
            
            # 将图像窗口去均值化
            winI -= win_mu
            
            # 计算拉普拉斯矩阵的值
            tvals = (1 + np.dot(np.dot(winI, np.linalg.inv(win_var)), winI.T)) / neb_size
            
            # 将矩阵数据填充到稀疏矩阵中
            for idx, row_idx in enumerate(win_inds):
                for jdx, col_idx in enumerate(win_inds):
                    row_inds.append(row_idx)
                    col_inds.append(col_idx)
                    vals.append(tvals[idx, jdx])
    
    # 转换为稀疏矩阵格式
    A = lil_matrix((img_size, img_size))
    for i, (row_idx, col_idx, val) in enumerate(zip(row_inds, col_inds, vals)):
        A[row_idx, col_idx] = val
    
    # 计算 A 的对角线值，并进行调整
    row_sums = np.array(A.sum(axis=1)).flatten()
    A.setdiag(row_sums - A.diagonal())
    
    return A

def softmatting(I, tmap, epsilon=1e-7, win_size=1, lambda_=0.001):
    """
    软渲染传导图（Soft Matting）
    
    :param I: 输入图像，double 类型
    :param tmap: 原始传导图
    :param epsilon: 防止数值不稳定的小常数
    :param win_size: 邻域窗口大小
    :param lambda_: 正则化参数
    :return: 精细化的传导图 tmap_ref
    """
    h, w, _ = I.shape
    img_size = w * h

    # 初始化权重矩阵 win_b
    win_b = np.zeros(img_size)
    for ci in range(h):
        for cj in range(w):
            if (ci - 8) % 15 == 0 and (cj - 8) % 15 == 0:
                win_b[ci * w + cj] = tmap[ci * w + cj]
    
    # 获取拉普拉斯矩阵 A
    A = get_laplacian(I, epsilon, win_size)
    
    # 构造稀疏对角矩阵 D
    D = spdiags(win_b, 0, img_size, img_size)
    
    # 解线性系统计算精细化传导图
    x = linalg.spsolve(A + lambda_ * D, lambda_ * win_b * win_b)
    
    # 将结果重新塑形为图像
    tmap_ref = np.clip(np.reshape(x, (h, w)), 0, 1)
    
    return tmap_ref

# 示例用法
if __name__ == "__main__":
    # 读取图像并将其转换为 double 类型
    I = cv2.imread('foggy1.jpg').astype(np.float64) / 255.0
    
    # 假设原始传导图 tmap 已经获得（可以是估计值）
    tmap = np.ones(I.shape[:2]) * 0.8  # 示例传导图
    
    # 获取精细化的传导图
    tmap_ref = softmatting(I, tmap)
    
    # 可视化结果
    cv2.imshow('Refined Transmission Map', tmap_ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
