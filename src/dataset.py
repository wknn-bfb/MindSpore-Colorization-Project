import os
import cv2
import numpy as np
import glob
from skimage import color
import mindspore.dataset as ds


class ColorizationDataset:
    """
    通用上色数据集加载器
    增强点：加入随机裁剪、水平翻转数据增强，提升模型泛化能力。
    """

    def __init__(self, data_root, dataset_name="coco", split="train", image_size=256):
        self.data_root = data_root
        self.image_size = image_size
        self.split = split
        self.image_paths = []

        # --- 1. 智能文件搜索 ---
        # 兼容 Windows/Linux 路径分隔符
        data_root = os.path.normpath(data_root)
        print(f"Scanning files for {dataset_name.upper()} dataset in {data_root}...")

        if dataset_name.lower() == "ncd":
            search_pattern = os.path.join(data_root, "**", "*.[jJ][pP][gG]")
            self.image_paths = glob.glob(search_pattern, recursive=True)
            self.image_paths += glob.glob(os.path.join(data_root, "**", "*.[pP][nN][gG]"), recursive=True)
        else:
            # 扁平结构
            if os.path.exists(data_root):
                self.image_paths = [os.path.join(data_root, x) for x in os.listdir(data_root)
                                    if x.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            else:
                print(f"Error: Path {data_root} does not exist.")

        self.image_paths.sort()

        # 简单的 Train/Val 划分 (9:1)
        if split != "test":
            total_len = len(self.image_paths)
            val_len = int(total_len * 0.1)
            if split == "train":
                self.image_paths = self.image_paths[val_len:]
            elif split == "val":
                self.image_paths = self.image_paths[:val_len]

        print(f"[{split.upper()}] Loaded {len(self.image_paths)} images.")

    def __getitem__(self, index):
        # 1. 读取图像 (RGB)
        img_path = self.image_paths[index]
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Image is None")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # 健壮性：读取失败返回纯黑图，避免训练崩溃
            print(f"Warning: Failed to load {img_path}: {e}")
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # 2. 数据增强 (仅训练集)
        if self.split == 'train':
            img = self._augment(img)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))

        # 3. RGB -> Lab
        img_float = img.astype(np.float32) / 255.0
        # skimage 的 rgb2lab 返回 L:0-100, a/b:-128-127
        img_lab = color.rgb2lab(img_float)

        # 4. 归一化到 [-1, 1]
        img_l = img_lab[:, :, 0:1]
        img_l = (img_l - 50.0) / 50.0  # L centered

        img_ab = img_lab[:, :, 1:3]
        img_ab = img_ab / 110.0  # ab normalized roughly

        # 5. 模拟用户提示
        if self.split == 'train':
            hint_ab, mask = self.simulate_user_hints(img_ab)
        else:
            hint_ab = np.zeros_like(img_ab)
            mask = np.zeros((self.image_size, self.image_size, 1), dtype=np.float32)

        # 6. HWC -> CHW
        return (img_l.transpose(2, 0, 1).astype(np.float32),
                img_ab.transpose(2, 0, 1).astype(np.float32),
                hint_ab.transpose(2, 0, 1).astype(np.float32),
                mask.transpose(2, 0, 1).astype(np.float32))

    def _augment(self, img):
        """数据增强：随机裁剪 + 随机翻转"""
        h, w, _ = img.shape
        # 1. Resize 稍微大一点 (例如 286x286)
        resize_h = int(self.image_size * 1.12)
        resize_w = int(self.image_size * 1.12)
        img = cv2.resize(img, (resize_w, resize_h))

        # 2. Random Crop
        dy = np.random.randint(0, resize_h - self.image_size + 1)
        dx = np.random.randint(0, resize_w - self.image_size + 1)
        img = img[dy:dy + self.image_size, dx:dx + self.image_size, :]

        # 3. Random Horizontal Flip
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)

        return img

    def __len__(self):
        return len(self.image_paths)

    def simulate_user_hints(self, img_ab):
        h, w, c = img_ab.shape
        hint_ab = np.zeros_like(img_ab)
        mask = np.zeros((h, w, 1), dtype=np.float32)

        # 策略：混合无提示(自动上色)和有提示
        rand_val = np.random.rand()
        if rand_val < 0.2:
            return hint_ab, mask  # 无提示

        # 随机点数 1~10
        num_hints = np.random.randint(1, 11)

        for _ in range(num_hints):
            # 几何分布采样位置，偏向物体中心（简单模拟）或完全随机
            rx = np.random.randint(0, w)
            ry = np.random.randint(0, h)

            # 随机笔触大小 (3x3 到 9x9)
            patch_size = np.random.randint(1, 5) * 2 + 1
            half_p = patch_size // 2

            x_start = max(0, rx - half_p)
            x_end = min(w, rx + half_p + 1)
            y_start = max(0, ry - half_p)
            y_end = min(h, ry + half_p + 1)

            hint_ab[y_start:y_end, x_start:x_end, :] = img_ab[y_start:y_end, x_start:x_end, :]
            mask[y_start:y_end, x_start:x_end, :] = 1.0

        return hint_ab, mask


def create_dataset(data_root, dataset_name="coco", split="train", batch_size=4, device_num=1, rank_id=0):
    dataset_generator = ColorizationDataset(data_root, dataset_name=dataset_name, split=split)

    # 容错：如果数据集为空，抛出清晰错误
    if len(dataset_generator) == 0:
        raise ValueError(f"Dataset is empty. Check path: {data_root}")

    ds_data = ds.GeneratorDataset(dataset_generator,
                                  column_names=["l_data", "ab_gt", "ab_hint", "mask"],
                                  shuffle=(split == "train"),
                                  num_shards=device_num,
                                  shard_id=rank_id,
                                  num_parallel_workers=1)

    ds_data = ds_data.batch(batch_size, drop_remainder=(split == "train"))
    return ds_data