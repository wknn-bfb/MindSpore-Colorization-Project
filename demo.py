import os
import cv2
import numpy as np
from skimage import color
import tkinter as tk
from tkinter import colorchooser

import mindspore as ms
from mindspore import Tensor, ops
from src.model import UNetGenerator
from src.utils import lab_to_rgb

# ================= 配置 =================
CKPT_PATH = 'checkpoints/net_g_40.ckpt'
IMG_PATH = 'data/demo_imgs/car.jpg'  # 修改这里更换测试图
IMG_SIZE = 256
OUTPUT_DIR = 'results/demo_showcase'  # 结果保存路径
# =======================================

# 全局变量
user_hints = []  # 存储提示点 [(x, y, a, b), ...]
img_l_norm = None  # L 通道数据
img_bgr_resized = None  # 原图用于显示
net = None  # 模型
current_display_img = None  # 当前展示的完整拼接图
save_counter = 1  # 【新增】手动保存计数器，从1开始


def get_ab_from_hex(hex_color):
    """把 Hex 颜色 (#FF0000) 转为归一化的 ab 值"""
    if not hex_color: return 0.0, 0.0

    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    rgb = np.array([[[r, g, b]]], dtype=np.uint8)

    # RGB -> Lab
    lab = color.rgb2lab(rgb.astype(np.float32) / 255.0)

    # 归一化 (根据 dataset.py 逻辑)
    a_norm = lab[0, 0, 1] / 110.0
    b_norm = lab[0, 0, 2] / 110.0
    return a_norm, b_norm


def run_inference():
    """根据当前的 user_hints 重新运行模型"""
    global img_l_norm, net

    # 1. 构造 Hint 和 Mask
    h, w = IMG_SIZE, IMG_SIZE
    t_hint = np.zeros((1, 2, h, w), dtype=np.float32)
    t_mask = np.zeros((1, 1, h, w), dtype=np.float32)

    # 2. 填充提示点
    for (cx, cy, val_a, val_b) in user_hints:
        patch_size = 9  # 笔触大小
        half = patch_size // 2
        y_min = max(0, cy - half)
        y_max = min(h, cy + half)
        x_min = max(0, cx - half)
        x_max = min(w, cx + half)

        t_mask[:, :, y_min:y_max, x_min:x_max] = 1.0
        t_hint[:, 0, y_min:y_max, x_min:x_max] = val_a
        t_hint[:, 1, y_min:y_max, x_min:x_max] = val_b

    # 3. 构造输入 Tensor
    t_l = Tensor(img_l_norm[None, None, ...], ms.float32)
    t_hint = Tensor(t_hint, ms.float32)
    t_mask = Tensor(t_mask, ms.float32)

    x_input = ops.concat((t_l, t_hint, t_mask), axis=1)

    # 4. 推理
    pred_ab = net(x_input)

    # 5. 转回 RGB
    pred_ab_np = pred_ab.asnumpy()[0]
    rgb_out = lab_to_rgb(img_l_norm[None, ...], pred_ab_np)

    return rgb_out


def mouse_callback(event, x, y, flags, param):
    """鼠标点击事件回调"""
    global user_hints

    if event == cv2.EVENT_LBUTTONDOWN:
        # 处理点击坐标：允许点击左图或右图
        # 如果点击了右边的结果图 (x >= 256)，则减去偏移量，映射回原图坐标
        real_x = x
        if x >= IMG_SIZE:
            real_x = x - IMG_SIZE

        # 防止越界
        if real_x < 0 or real_x >= IMG_SIZE or y < 0 or y >= IMG_SIZE:
            return

        print(f"👉 选中坐标: ({real_x}, {y})，正在打开取色器...")

        # 1. 打开系统取色板 (Tkinter)
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        color_code = colorchooser.askcolor(title="Choose Color")[1]
        root.destroy()

        if color_code:
            print(f"🎨 用户选择颜色: {color_code}")

            # 2. 计算 ab 值并存入历史
            val_a, val_b = get_ab_from_hex(color_code)
            user_hints.append((real_x, y, val_a, val_b))

            # 3. 触发更新
            update_display()


def update_display():
    """刷新显示窗口"""
    global current_display_img

    # 1. 运行推理
    res_rgb = run_inference()

    # 2. 转为 BGR 用于 OpenCV 显示
    res_bgr = cv2.cvtColor((res_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 3. 在结果图上画圈标记点击位置 (显示用户的操作痕迹)
    for (hx, hy, _, _) in user_hints:
        # 画在右图上
        cv2.circle(res_bgr, (hx, hy), 6, (0, 255, 0), 1)  # 绿色空心圆
        cv2.circle(res_bgr, (hx, hy), 2, (0, 255, 0), -1)  # 绿色实心点

    # 4. 拼接显示：左边原图 | 右边结果
    current_display_img = np.hstack((img_bgr_resized, res_bgr))

    cv2.imshow("MindSpore Interactive Demo (Press 'r' to reset, 's' to save, 'q' to quit)", current_display_img)


def save_current_result(is_auto=False):
    """
    保存结果逻辑
    is_auto: True表示自动保存demo0, False表示手动保存demo1,2...
    """
    global save_counter

    if current_display_img is not None:
        base_name = os.path.splitext(os.path.basename(IMG_PATH))[0]

        if is_auto:
            # 自动保存为 demo0
            suffix = "_demo0.png"
            print("🤖 [自动存档] 检测到无提示初始化，正在保存 demo0...")
        else:
            # 手动保存为 demo1, demo2...
            suffix = f"_demo{save_counter}.png"
            save_counter += 1

        save_name = f"{base_name}{suffix}"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        cv2.imwrite(save_path, current_display_img)
        print(f"💾 已保存交互窗口至: {save_path}")


# ================= 主程序 =================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 初始化 MindSpore
    print("🖥️  初始化模型 (CPU)...")
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    net = UNetGenerator(input_nc=4, output_nc=2)
    try:
        param_dict = ms.load_checkpoint(CKPT_PATH)
        ms.load_param_into_net(net, param_dict)
        net.set_train(False)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()

    # 2. 准备图片
    if not os.path.exists(IMG_PATH):
        print(f"❌ 找不到图片: {IMG_PATH}")
        exit()

    raw_img = cv2.imread(IMG_PATH)
    raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
    img_bgr_resized = raw_img.copy()  # 备份一份用于显示

    # 预处理 L 通道
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    img_lab = color.rgb2lab(img_rgb.astype(np.float32) / 255.0)
    img_l_norm = (img_lab[:, :, 0] - 50.0) / 50.0  # 全局变量

    # 3. 启动 GUI
    print("🚀 交互界面已启动！")
    print(f"   当前处理图片: {os.path.basename(IMG_PATH)}")
    print("   [操作指南]")
    print("   🖱️  点击图片任意位置 (左图/右图皆可) -> 选择颜色")
    print("   ⌨️  'r' -> 重置所有提示点")
    print("   ⌨️  's' -> 保存当前交互窗口 (自动递增 demo1, demo2...)")
    print("   ⌨️  'q' -> 退出")

    win_name = "MindSpore Interactive Demo (Press 'r' to reset, 's' to save, 'q' to quit)"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback)

    # 4. 初始显示 + 自动保存 demo0
    update_display()
    # 【新增】启动时自动保存无提示版本为 demo0
    save_current_result(is_auto=True)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == ord('r'):  # Reset
            print("🔄 重置提示点")
            user_hints = []
            update_display()
        elif key == ord('s'):  # Save
            # 【新增】手动保存，逻辑是 demo1, demo2...
            save_current_result(is_auto=False)

    cv2.destroyAllWindows()