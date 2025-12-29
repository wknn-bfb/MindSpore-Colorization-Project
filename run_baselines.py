import argparse
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorizers import *  # 导入同级目录下的 colorizers 包


def main():
    parser = argparse.ArgumentParser(description="运行两条 Baseline 模型并保存结果")

    # --- 1. 路径配置 ---
    # 脚本运行在 baselines/ 目录下，所以要用 ../ 返回上一级
    parser.add_argument('-i', '--img_dir', type=str, default='../data/test_set',
                        help='输入图片文件夹 (RGB)')

    # 输出路径分开配置
    parser.add_argument('--out_dir_eccv', type=str, default='../results/baseline1',
                        help='ECCV16 结果保存路径')
    parser.add_argument('--out_dir_siggraph', type=str, default='../results/baseline2',
                        help='SIGGRAPH17 结果保存路径')

    # 权重文件路径
    parser.add_argument('--ckpt_eccv', type=str,
                        default='../checkpoints/colorization_release_v2-9b330a0b.pth',
                        help='ECCV16 权重文件路径')
    parser.add_argument('--ckpt_siggraph', type=str,
                        default='../checkpoints/siggraph17-df00044c.pth',
                        help='SIGGRAPH17 权重文件路径')

    parser.add_argument('--use_gpu', action='store_true', help='是否使用 GPU 加速')

    opt = parser.parse_args()

    # --- 2. 检查路径有效性 ---
    if not os.path.exists(opt.img_dir):
        print(f"错误：找不到输入文件夹 {opt.img_dir}")
        return

    if not os.path.exists(opt.ckpt_eccv):
        print(f"错误：找不到 ECCV16 权重 {opt.ckpt_eccv}")
        return

    if not os.path.exists(opt.ckpt_siggraph):
        print(f"错误：找不到 SIGGRAPH17 权重 {opt.ckpt_siggraph}")
        return

    # 自动创建输出目录
    os.makedirs(opt.out_dir_eccv, exist_ok=True)
    os.makedirs(opt.out_dir_siggraph, exist_ok=True)

    # --- 3. 加载模型 (Load Models) ---
    # 初始化模型结构 (pretrained=False 防止自动下载)
    colorizer_eccv16 = eccv16(pretrained=False).eval()
    colorizer_siggraph17 = siggraph17(pretrained=False).eval()

    # 加载本地权重
    colorizer_eccv16.load_state_dict(torch.load(opt.ckpt_eccv))
    colorizer_siggraph17.load_state_dict(torch.load(opt.ckpt_siggraph))
    print("模型权重加载成功！")

    # 移动到 GPU
    if opt.use_gpu and torch.cuda.is_available():
        colorizer_eccv16.cuda()
        colorizer_siggraph17.cuda()
        print("使用 GPU 推理")
    else:
        print("使用 CPU 推理")

    # --- 4. 批量处理循环 ---
    # 获取所有图片
    img_files = [f for f in os.listdir(opt.img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(img_files)} 张测试图片，开始处理...")

    for img_name in tqdm(img_files):
        full_path = os.path.join(opt.img_dir, img_name)

        # A. 加载与预处理
        # load_img 读取 RGB，preprocess_img 自动转换为 Lab 并提取 L 通道
        img = load_img(full_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

        if opt.use_gpu and torch.cuda.is_available():
            tens_l_rs = tens_l_rs.cuda()

        # B. 推理 (Inference)
        # 自动上色 (ECCV16)
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        # 交互上色-自动模式 (SIGGRAPH17)
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

        # C. 保存结果 (Save)
        name_base = os.path.splitext(img_name)[0]

        # 保存 ECCV16
        save_path_eccv = os.path.join(opt.out_dir_eccv, f"{name_base}_eccv16.png")
        plt.imsave(save_path_eccv, out_img_eccv16)

        # 保存 SIGGRAPH17
        save_path_siggraph = os.path.join(opt.out_dir_siggraph, f"{name_base}_siggraph17.png")
        plt.imsave(save_path_siggraph, out_img_siggraph17)

    print("\n 所有 Baseline 处理完成！")
    print(f" ECCV16 结果: {os.path.abspath(opt.out_dir_eccv)}")
    print(f" SIGGRAPH17 结果: {os.path.abspath(opt.out_dir_siggraph)}")


if __name__ == '__main__':
    main()