import numpy as np
import torch
import torch.nn.functional as F

# 计算局部相似性
def compute_similarity(I_x, I_y, alpha=50):
    """ 计算两个图像块之间的相似性 """
    I_x = torch.tensor(I_x, dtype=torch.float32)
    I_y = torch.tensor(I_y, dtype=torch.float32)
    return torch.exp(-torch.norm(I_x - I_y) ** 2 / alpha)

# 对每个像素计算局部相似性
def compute_local_self_similarity(image, patch_size=5, alpha=50):
    h, w = image.shape
    similarity_map = torch.zeros(h, w)
    
    # 遍历每个像素点
    for i in range(patch_size//2, h-patch_size//2):
        for j in range(patch_size//2, w-patch_size//2):
            patch_x = image[i - patch_size//2 : i + patch_size//2 + 1, j - patch_size//2 : j + patch_size//2 + 1]
            
            # 计算当前点与周围区域的相似性
            max_similarity = 0
            for di in range(-patch_size//2, patch_size//2):
                for dj in range(-patch_size//2, patch_size//2):
                    patch_y = image[i + di - patch_size//2 : i + di + patch_size//2 + 1, 
                                     j + dj - patch_size//2 : j + dj + patch_size//2 + 1]
                    similarity = compute_similarity(patch_x, patch_y, alpha)
                    max_similarity = max(max_similarity, similarity)
            
            similarity_map[i, j] = max_similarity
    
    return similarity_map

def cartesian_to_polar(x, y, xc, yc):
    """ 将直角坐标转换为极坐标 """
    rho = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    theta = np.arctan2(y - yc, x - xc)
    return rho, theta

def extract_max_patch_value(image, patch_size=5):
    """ 对图像进行分块并提取每个小块的最大值 """
    h, w = image.shape
    descriptors = []

    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            max_value = patch.max()
            descriptors.append(max_value)
    
    return torch.tensor(descriptors)

def extract_local_self_similarity_descriptor(image, patch_size=5, alpha=50):
    similarity_map = compute_local_self_similarity(image, patch_size, alpha)
    descriptors = extract_max_patch_value(similarity_map, patch_size)
    return descriptors

# 提取图像块
def extract_patch(image, top_left, patch_size=(80, 120)):
    x, y = top_left
    w, h = patch_size
    patch = image[:, y:y+h, x:x+w]
    return torch.tensor(patch, dtype=torch.float32)

# 匹配图像块
def match_patches(image1, image2, patch_size=(80, 120)):
    patch1 = extract_patch(image1, (0, 0), patch_size)
    patch2 = extract_patch(image2, (0, 0), patch_size)
    similarity = compute_similarity(patch1, patch2)
    return similarity

def find_best_match(image1, image2, patch_size=(80, 120)):
    h, w = image1.shape[1], image1.shape[2]
    best_similarity = -1
    best_match = None
    
    # 遍历图像中的每个矩形框
    for i in range(0, h - patch_size[1] + 1, patch_size[1]):
        for j in range(0, w - patch_size[0] + 1, patch_size[0]):
            patch1 = extract_patch(image1, (j, i), patch_size)
            
            for ii in range(0, h - patch_size[1] + 1, patch_size[1]):
                for jj in range(0, w - patch_size[0] + 1, patch_size[0]):
                    patch2 = extract_patch(image2, (jj, ii), patch_size)
                    similarity = compute_similarity(patch1, patch2)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = ((j, i), (jj, ii))  # 返回匹配的区域
    return best_match, best_similarity

# 计算匹配的显著性
def calculate_significance(best_similarities):
    avg_match = np.mean(best_similarities)
    std_match = np.std(best_similarities)
    
    significance_scores = [(similarity - avg_match) / std_match if std_match != 0 else 0 
                           for similarity in best_similarities]
    
    return significance_scores

def find_best_match_with_significance(best_matches, best_similarities):
    significance_scores = calculate_significance(best_similarities)
    
    max_significance_index = np.argmax(significance_scores)
    
    return best_matches[max_significance_index], significance_scores[max_significance_index]

# 主要功能：提取描述子并进行目标匹配与显著性分析
def extract_and_match(images_np, patch_size=(80, 120)):
    best_matches_all = []
    best_similarities_all = []
    
    # 遍历所有图像，假设第一张是基准图像
    image1 = images_np[0]  # 第一张图像
    images = images_np[1:]  # 其他图像

    # 计算自相似描述子并进行匹配
    for image2 in images:
        best_match, best_similarity = find_best_match(image1, image2, patch_size)
        best_matches_all.append(best_match)
        best_similarities_all.append(best_similarity)
    
    # 计算显著性并筛选出最佳匹配
    best_match, significance_score = find_best_match_with_significance(best_matches_all, best_similarities_all)
    
    return best_match, significance_score

# 使用示例
images_np = np.random.rand(5, 3, 224, 224)  # 示例数据，5张224x224的RGB图像
best_match, significance_score = extract_and_match(images_np)
print("Best match:", best_match)
print("Significance score:", significance_score)
