import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, load_checkpoint
import mindspore.common.dtype as mstype
import traceback

class VGG16FeatureExtractor(nn.Cell):
    """
    VGG16_BN 特征提取器
    """
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        
        # Slice 1: 2 Convs
        self.slice1 = nn.SequentialCell([
            nn.Conv2d(3, 64, 3, padding=1, pad_mode='pad', has_bias=False), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, pad_mode='pad', has_bias=False), 
            nn.BatchNorm2d(64), nn.ReLU()
        ])
        # Slice 2: 2 Convs
        self.slice2 = nn.SequentialCell([
            nn.MaxPool2d(2, 2, pad_mode='valid'),
            nn.Conv2d(64, 128, 3, padding=1, pad_mode='pad', has_bias=False), 
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, pad_mode='pad', has_bias=False), 
            nn.BatchNorm2d(128), nn.ReLU()
        ])
        # Slice 3: 3 Convs
        self.slice3 = nn.SequentialCell([
            nn.MaxPool2d(2, 2, pad_mode='valid'),
            nn.Conv2d(128, 256, 3, padding=1, pad_mode='pad', has_bias=False), 
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, pad_mode='pad', has_bias=False), 
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, pad_mode='pad', has_bias=False), 
            nn.BatchNorm2d(256), nn.ReLU()
        ])

        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return h1, h2, h3

class UserGuidedLoss(nn.Cell):
    def __init__(self, vgg_ckpt_path=None, lambda_l1=100.0, lambda_gan=1.0, lambda_perceptual=10.0):
        super(UserGuidedLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_gan = lambda_gan
        self.lambda_perceptual = lambda_perceptual
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss()
        self.mean = Tensor([0.485, 0.456, 0.406], mstype.float32).view(1, 3, 1, 1)
        self.std = Tensor([0.229, 0.224, 0.225], mstype.float32).view(1, 3, 1, 1)

        self.vgg = None
        if vgg_ckpt_path:
            print(f"Initializing VGG16_BN for Perceptual Loss from {vgg_ckpt_path}...")
            try:
                self.vgg = VGG16FeatureExtractor()
                ckpt_params = load_checkpoint(vgg_ckpt_path)
                
                # ====================================================
                # 终极方案：直接赋值 (Direct Assignment)
                # 不依赖 load_param_into_net，不依赖参数名匹配
                # ====================================================
                
                # 1. 准备 Checkpoint 数据
                # ----------------------------------------------------
                # 筛选 layers.* 并按数字索引排序
                valid_keys = [k for k in ckpt_params.keys() if k.startswith('layers.')]
                def get_layer_idx(key):
                    try: return int(key.split('.')[1])
                    except: return 9999
                sorted_keys = sorted(valid_keys, key=get_layer_idx)
                
                # 分离 Conv 和 BN 的参数数据
                ckpt_conv_weights = []
                # BN 参数需要按组存储: (gamma, beta, mean, var)
                # 我们使用字典按 layer_idx 归组
                ckpt_bn_groups = {} 
                
                for k in sorted_keys:
                    val = ckpt_params[k]
                    idx = get_layer_idx(k)
                    
                    if val.ndim == 4: # Conv Weight
                        ckpt_conv_weights.append(val)
                    else: # BN Params
                        if idx not in ckpt_bn_groups:
                            ckpt_bn_groups[idx] = {}
                        
                        if 'gamma' in k: ckpt_bn_groups[idx]['gamma'] = val
                        elif 'beta' in k: ckpt_bn_groups[idx]['beta'] = val
                        elif 'moving_mean' in k: ckpt_bn_groups[idx]['mean'] = val
                        elif 'moving_variance' in k: ckpt_bn_groups[idx]['var'] = val

                # 将 BN 组按 index 排序转为列表
                sorted_bn_indices = sorted(ckpt_bn_groups.keys())
                ckpt_bn_list = [ckpt_bn_groups[i] for i in sorted_bn_indices]

                print(f"DEBUG: Checkpoint has {len(ckpt_conv_weights)} Conv weights.")
                print(f"DEBUG: Checkpoint has {len(ckpt_bn_list)} BN layers.")

                # 2. 遍历网络层并直接赋值
                # ----------------------------------------------------
                conv_ptr = 0
                bn_ptr = 0
                loaded_count = 0
                
                # 递归遍历所有子 Cell
                for _, cell in self.vgg.cells_and_names():
                    if isinstance(cell, nn.Conv2d):
                        if conv_ptr < len(ckpt_conv_weights):
                            # 直接赋值！
                            cell.weight.set_data(ckpt_conv_weights[conv_ptr])
                            conv_ptr += 1
                            loaded_count += 1
                    
                    elif isinstance(cell, nn.BatchNorm2d):
                        if bn_ptr < len(ckpt_bn_list):
                            bn_data = ckpt_bn_list[bn_ptr]
                            # 直接赋值！
                            if 'gamma' in bn_data: cell.gamma.set_data(bn_data['gamma'])
                            if 'beta' in bn_data: cell.beta.set_data(bn_data['beta'])
                            if 'mean' in bn_data: cell.moving_mean.set_data(bn_data['mean'])
                            if 'var' in bn_data: cell.moving_variance.set_data(bn_data['var'])
                            bn_ptr += 1
                            loaded_count += 4 # BN 有4个参数

                print(f"DEBUG: Directly assigned {conv_ptr} Convs and {bn_ptr} BNs.")

                # 3. 验证
                if conv_ptr >= 7 and bn_ptr >= 7:
                    print("✅ VGG16_BN loaded successfully (Direct Assignment).")
                else:
                    print(f"⚠️ VGG Partial load. Only loaded {conv_ptr} Convs.")

                self.vgg.set_train(False)

            except Exception as e:
                print(f"⚠️ Warning: Failed to load VGG: {e}")
                traceback.print_exc()
                self.vgg = None

    def get_gan_loss(self, pred, target_is_real):
        target = ops.ones_like(pred) if target_is_real else ops.zeros_like(pred)
        return self.bce_loss(pred, target)

    def construct(self, d_pred_real, d_pred_fake, g_pred_ab, real_ab, mask, real_rgb, fake_rgb):
        loss_d_real = self.get_gan_loss(d_pred_real, True)
        loss_d_fake = self.get_gan_loss(d_pred_fake, False)
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_g_gan = self.get_gan_loss(d_pred_fake, True)
        diff = self.l1_loss(g_pred_ab, real_ab)
        weight_map = 1.0 + (mask * 50.0)
        loss_g_l1 = (diff * weight_map).mean()
        loss_g_perceptual = Tensor(0.0, mstype.float32)
        if self.vgg is not None:
            real_norm = (real_rgb - self.mean) / self.std
            fake_norm = (fake_rgb - self.mean) / self.std
            r_h1, r_h2, r_h3 = self.vgg(real_norm)
            f_h1, f_h2, f_h3 = self.vgg(fake_norm)
            loss_g_perceptual = (self.mse_loss(f_h1, r_h1) + self.mse_loss(f_h2, r_h2) + self.mse_loss(f_h3, r_h3)) / 3.0
        loss_g = (loss_g_gan * self.lambda_gan) + (loss_g_l1 * self.lambda_l1) + (loss_g_perceptual * self.lambda_perceptual)
        return loss_g, loss_d
