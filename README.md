# MindSpore-Colorization-Project

> 基于 MindSpore 的用户引导式图像上色实现  
> 交互点一下，灰图立刻出彩

## 🚀 快速开始

1. 克隆并下载权重（>100 MB，脚本自动拉取）
   ```bash
   git clone https://github.com/wknn-bfb/MindSpore-Colorization-Project.git
   cd MindSpore-Colorization-Project
   bash scripts/download_weights.sh      # Linux / macOS
   rem scripts\download_weights.bat      # Windows
   ```

2. 一键上色
   ```bash
   python demo.py
   ```
   如需修改上色图，请在demo.py中修改路径
   
   | 操作 | 效果 |
   |---|---|
   | 左键点击图片任意位置 | 弹出取色器，选择颜色后实时上色 |
   | `r` | 重置所有提示点 |
   | `s` | 保存当前交互窗口（自动递增 demo1, demo2...） |
   | `q` | 退出 |

3. 训练
   训练全部基于华为云平台实现，相关代码位于src文件夹下。
4. 评估
   
   ```bash
   python evaluate.py --pred_dir results/our --gt_dir data/test_set
   ```

## 📂 目录一览

```
├── baselines/          # ECCV16 & SigGraph17 
├── checkpoints/        # 下载的 *.ckpt / *.pth（git-ignored）
├── data/               #  demo_imgs 可提交；train/test 自行准备
├── results/            # 输出目录（git-ignored）
├── scripts/            # 权重下载脚本
├── src/                # 模型、损失、数据集、训练逻辑
├── demo.py             # 单图推理 / 交互 GUI
├── run_baselines.py    # 批量跑基线
├── run_our_model.py    # 批量跑我们的模型
└── requirements.txt    # pip 一键装依赖
```

## ⚙️ 依赖

- Python ≥ 3.8
- MindSpore ≥ 2.0
- OpenCV-Python
- scikit-image
- tqdm, Pillow, numpy

一键安装  
```bash
pip install -r requirements.txt
```

## 📄 更多信息

技术细节、实验数据与完整报告见仓库内 [`report.docx`](report.docx)。  
欢迎提 Issue / PR 一起改进！

## 🤝 Acknowledgement

- 原始 U-Net & PatchGAN 设计：Olaf Ronneberger et al.  
- 感知损失 VGG 权重：PyTorch 官方预训练模型  
- 基线代码参考：ECCV16 Colorization、SIGGRAPH17 Colorization  
- 华为 MindSpore 团队提供的算子支持与图模式优化建议  
- 数据集：COCO-2017

## 📄 许可证

MIT © wknn-bfb
```
