import numpy as np
import cv2
from skimage import color
import matplotlib.pyplot as plt


def lab_to_rgb(l_img, ab_img):
    """
    将模型输出的 Tensor (归一化后的 Lab) 转回 RGB 图片用于显示
    l_img: (1, H, W) range [-1, 1]
    ab_img: (2, H, W) range [-1, 1]
    """
    # 1. 反归一化
    l_img = (l_img + 1.0) * 50.0
    ab_img = ab_img * 110.0

    # 2. 拼接 L 和 ab
    # 需要转回 HWC 格式
    l_img = l_img.transpose(1, 2, 0)
    ab_img = ab_img.transpose(1, 2, 0)
    lab_img = np.concatenate((l_img, ab_img), axis=2)

    # 3. Lab -> RGB
    # 可能会有越界警告，忽略即可
    rgb_img = color.lab2rgb(lab_img.astype(np.float64))
    return rgb_img


def visualize_batch(l_data, ab_gt, ab_hint, mask, save_path="debug_vis.png"):
    """
    可视化一个 Batch 的数据，用于检查 Dataset 是否写对
    """
    # 取 Batch 中的第一张图
    l = l_data[0].asnumpy()
    ab = ab_gt[0].asnumpy()
    hint = ab_hint[0].asnumpy()
    m = mask[0].asnumpy()

    # 还原图像
    img_gt = lab_to_rgb(l, ab)
    img_hint = lab_to_rgb(l, hint)  # 只有提示点有颜色，其他是灰的

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_gt)
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_hint)
    plt.title("User Hints Input")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(m[0], cmap='gray')
    plt.title("Hint Mask")
    plt.axis('off')

    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")