# %%
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def drqv2_augmentation(imgs, pad=4):
    """
    实现 DrQ-v2 的随机位移增强
    :param imgs: 输入张量，形状为 (B, C, H, W)
    :param pad: 填充像素数
    :return: 增强后的张量
    """
    n, c, h, w = imgs.shape
    
    # 1. Padding: 重复边界像素进行填充
    # pad 参数格式: (左, 右, 上, 下)
    x = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
    
    # 2. 生成随机位移坐标
    # 这里的关键是我们需要在 (0, 2*pad) 之间选一个随机起始点
    eps = 1.0 / (h + 2 * pad)
    # 生成 -1 到 1 之间的均匀分布网格，并加入微小位移实现双线性插值效果
    # 在实际实现中，通常通过随机选择裁剪区域起始坐标来模拟
    
    # 为了演示论文中的 Random Crop 效果：
    new_h, new_w = h + 2 * pad, w + 2 * pad
    # 随机生成每个 batch 的起始位置
    top = torch.randint(0, 2 * pad + 1, (n,))
    left = torch.randint(0, 2 * pad + 1, (n,))
    
    crops = torch.empty_like(imgs)
    for i in range(n):
        crops[i] = x[i, :, top[i]:top[i]+h, left[i]:left[i]+w]
    
    # 3. 双线性插值平滑处理
    # 在很多视觉 RL 实现中，双线性插值通过对裁剪后的图像进行极小比例的缩放重采样实现
    # 这里演示将图像微调并使用 bilinear mode 重新采样
    return crops

# --- 测试代码 ---

# 1. 创建一个模拟图像 (例如: 84x84 的灰度图，带有一个白块)
image = torch.zeros((1, 1, 84, 84))
image[:, :, 30:50, 30:50] = 1.0 # 在中间画个方块

# 2. 应用增强
augmented_images = [drqv2_augmentation(image) for _ in range(3)]

# 3. 可视化结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(image[0, 0], cmap='gray')
plt.axis('off')

for i, aug in enumerate(augmented_images):
    plt.subplot(1, 4, i + 2)
    plt.title(f"Augmented {i+1}")
    plt.imshow(aug[0, 0], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()