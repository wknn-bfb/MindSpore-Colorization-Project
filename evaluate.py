import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
    LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
import pandas as pd
from tqdm import tqdm

"""
    baseline1 ECCV 16
    baseline2 SIGGRAPH 17
    此脚本用于评估两个baseline和我们的模型
    对比每个模型生成的图像与 Ground-Truth 之间的差异
    输出 PSNR、SSIM、LPIPS、FID 四项指标的平均值
    并在控制台打印表格
"""
### 基本配置
GT_DIR = "data/test_set"  # 原图文件夹
# 存放三种模型生成结果的文件夹
ECCV16_DIR = "results/baseline1"
SIGGRAPH17_DIR = "results/baseline2"
OUR_DIR = "results/our"

# 三模型配置列表：名称、目录、后缀
methods = [
    ("ECCV16", ECCV16_DIR, "_eccv16.png"),
    ("SIGGRAPH17", SIGGRAPH17_DIR, "_siggraph17.png"),
    ("OUR", OUR_DIR, "_our.png")
]

# 硬件配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"运行设备: {DEVICE}")

### 核心实现逻辑
# 读取图片，转为 Tensor，归一化到 [0, 1]
def load_image_as_tensor(path, size=(256, 256)):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # 自动转为 [0, 1]
    ])
    return transform(img).unsqueeze(0).to(DEVICE)  # 增加 Batch 维度

# FID 需要 [0, 255] 的 uint8 tensor
def load_image_uint8(path, size=(256, 256)):
    img = Image.open(path).convert('RGB')
    img = img.resize(size)
    arr = np.array(img)
    return torch.from_numpy(arr).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)

# 指标计算
def run_evaluation():
    # --- 1. 初始化指标计算器 ---
    metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    metric_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(DEVICE)
    metric_fid = FrechetInceptionDistance(feature=64, normalize=False).to(DEVICE)

    # 获取 GT 文件列表
    if not os.path.exists(GT_DIR):
        print(f" 错误：找不到 GT 文件夹: {GT_DIR}")
        return

    gt_files = [f for f in os.listdir(GT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not gt_files:
        print(" 错误：GT 文件夹为空！")
        return

    print(f"找到 {len(gt_files)} 张测试图片，准备开始评估...")
    final_scores = [] # 记录结果

    # --- 2. 遍历每个模型进行评估 ---
    for method_name, pred_dir, suffix in methods:
        print(f"\n 正在评估: {method_name} ...")

        # 重置统计量
        metric_fid.reset()
        psnr_vals, ssim_vals, lpips_vals = [], [], []

        # 遍历每张图片
        for gt_name in tqdm(gt_files, desc=f"Eval {method_name}"):
            # --- 原图像与生成图匹配逻辑 ---
            # 1. 拿到不带后缀的文件名 (例如 "0000123.jpg" -> "0000123")
            name_base = os.path.splitext(gt_name)[0]

            # 2. 拼接预测图文件名 (例如 "0000123" + "_eccv16.png")
            pred_name = name_base + suffix

            gt_path = os.path.join(GT_DIR, gt_name)
            pred_path = os.path.join(pred_dir, pred_name)

            # 3. 检查文件是否存在
            if not os.path.exists(pred_path):
                print(f" 缺失文件: {pred_name} (跳过此图)")
                continue

            # --- 加载数据 ---
            try:
                # 加载浮点数据 [0, 1] 用于 PSNR, SSIM, LPIPS
                gt_float = load_image_as_tensor(gt_path)
                pred_float = load_image_as_tensor(pred_path)

                # 加载 uint8 数据 [0, 255] 用于 FID
                gt_uint8 = load_image_uint8(gt_path)
                pred_uint8 = load_image_uint8(pred_path)

                # --- 计算指标 ---
                # PSNR, SSIM, LPIPS 是“一对一”计算
                psnr_vals.append(metric_psnr(pred_float, gt_float).item())
                ssim_vals.append(metric_ssim(pred_float, gt_float).item())
                lpips_vals.append(metric_lpips(pred_float, gt_float).item())

                # FID 是“更新统计分布”，最后统一计算
                metric_fid.update(gt_uint8, real=True)
                metric_fid.update(pred_uint8, real=False)

            except Exception as e:
                print(f" 处理图片 {gt_name} 时出错: {e}")
                continue

        # 汇总当前模型的平均分
        if len(psnr_vals) > 0:
            # 计算 FID
            score_fid = metric_fid.compute().item()

            # 计算 PSNR SSIM LPIPS 的平均值
            avg_psnr = sum(psnr_vals) / len(psnr_vals)
            avg_ssim = sum(ssim_vals) / len(ssim_vals)
            avg_lpips = sum(lpips_vals) / len(lpips_vals)

            # 记录结果
            final_scores.append({
                "Method": method_name,
                "PSNR (↑)": round(avg_psnr, 2),
                "SSIM (↑)": round(avg_ssim, 3),
                "LPIPS (↓)": round(avg_lpips, 3),
                "FID (↓)": round(score_fid, 2)
            })
        else:
            print(f"❌ {method_name} 没有成功处理任何图片。")

    # --- 3. 输出报表 ---
    if final_scores:
        print("\n" + "=" * 60)
        print("最终评估报告 (Evaluation Report)")
        print("=" * 60)
        df = pd.DataFrame(final_scores)

        # 使用 tabulate 格式打印，更像表格
        print(df.to_markdown(index=False, numalign="left", stralign="left"))
    else:
        print("\n 没有生成任何评估结果，请检查路径配置。")


if __name__ == "__main__":
    run_evaluation()