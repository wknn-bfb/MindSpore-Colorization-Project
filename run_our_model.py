import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from skimage import color

import mindspore as ms
from mindspore import Tensor, ops

# 导入你的核心模块
from src.model import UNetGenerator
from src.utils import lab_to_rgb


def preprocess_image(img_path, target_size=(256, 256)):
    """
    读取并预处理图片：
    返回 4 个值: t_l, t_hint, t_mask, l_norm
    """
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Resize
    img_rs = cv2.resize(img, target_size)

    # 3. RGB -> Lab
    img_float = img_rs.astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_float)

    # 4. 提取 L 通道并归一化
    img_l = img_lab[:, :, 0]
    img_l_norm = (img_l - 50.0) / 50.0

    # 5. 构造 Tensor
    t_l = Tensor(img_l_norm[None, None, ...], ms.float32)
    t_hint = ops.zeros((1, 2, target_size[0], target_size[1]), ms.float32)
    t_mask = ops.zeros((1, 1, target_size[0], target_size[1]), ms.float32)

    return t_l, t_hint, t_mask, img_l_norm


def run(args):
    # --- 1. 强制 CPU ---
    print("🖥️  正在初始化 MindSpore (CPU模式)...")
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    if not os.path.exists(args.ckpt_path):
        print(f"❌ 错误：找不到权重文件 {args.ckpt_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. 加载模型 ---
    print(f"⏳ 正在加载模型: {args.ckpt_path} ...")
    # input_nc=4 说明模型期待 4 通道输入
    net = UNetGenerator(input_nc=4, output_nc=2)

    try:
        param_dict = ms.load_checkpoint(args.ckpt_path)
        ms.load_param_into_net(net, param_dict)
        net.set_train(False)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # --- 3. 遍历测试集 ---
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误：找不到输入文件夹 {args.input_dir}")
        return

    img_files = [f for f in os.listdir(args.input_dir)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    print(f"📋 找到 {len(img_files)} 张测试图片，开始推理...")

    for img_name in tqdm(img_files):
        img_path = os.path.join(args.input_dir, img_name)

        # A. 预处理
        t_l, t_hint, t_mask, l_norm = preprocess_image(img_path)

        if t_l is None:
            continue

        # B. 推理 (核心修复部分)
        # ========================================================
        # 错误原因：net() 只接收一个参数，之前传了三个
        # 修复方案：在传入前，先将 (L, Hint, Mask) 拼接成 4 通道张量
        # Shape: (1, 1, H, W) + (1, 2, H, W) + (1, 1, H, W) -> (1, 4, H, W)
        # ========================================================
        x_input = ops.concat((t_l, t_hint, t_mask), axis=1)

        # 现在只传这一个合并后的变量
        pred_ab = net(x_input)

        # C. 后处理
        pred_ab_np = pred_ab.asnumpy()[0]
        rgb_out = lab_to_rgb(l_norm[None, ...], pred_ab_np)

        # D. 保存
        name_base = os.path.splitext(img_name)[0]
        save_name = f"{name_base}_our.png"
        save_path = os.path.join(args.output_dir, save_name)

        img_bgr_out = cv2.cvtColor((rgb_out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr_out)

    print(f"🎉 所有结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1000张测试集
    parser.add_argument('--input_dir', type=str, default='data/test_set', help='测试集文件夹路径')
    parser.add_argument('--output_dir', type=str, default='results/our', help='结果保存路径')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/net_g_40.ckpt', help='模型权重路径')

    args = parser.parse_args()
    run(args)