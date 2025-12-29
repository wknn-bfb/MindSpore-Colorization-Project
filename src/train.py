import argparse
import os
import time
import numpy as np
import cv2
import mindspore as ms
from mindspore import nn, context, ops, save_checkpoint, load_checkpoint, load_param_into_net
from src.dataset import create_dataset
from src.model import UNetGenerator, PatchGANDiscriminator
from src.loss import UserGuidedLoss
from src.utils import lab_to_rgb

class GAN_TrainOneStepCell(nn.Cell):
    def __init__(self, net_g, net_d, optimizer_g, optimizer_d, loss_fn):
        super(GAN_TrainOneStepCell, self).__init__()
        self.net_g = net_g
        self.net_d = net_d
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.loss_fn = loss_fn
        self.grad = ops.GradOperation(get_by_list=True, sens_param=False)
        self.concat = ops.Concat(axis=1)
        self.zeros_like = ops.ZerosLike()
        self.depend = ops.Depend()

    def g_loss_func(self, l, hint, mask, ab_gt, real_rgb):
        g_input = self.concat((l, hint, mask))
        fake_ab = self.net_g(g_input)
        fake_rgb = self.concat((l, fake_ab))
        d_in_fake = self.concat((l, fake_ab))
        pred_fake = self.net_d(d_in_fake)
        dummy = self.zeros_like(pred_fake)
        loss_g, _ = self.loss_fn(dummy, pred_fake, fake_ab, ab_gt, mask, real_rgb, fake_rgb)
        return loss_g

    def d_loss_func(self, l, hint, mask, ab_gt, real_rgb):
        g_input = self.concat((l, hint, mask))
        fake_ab = self.net_g(g_input)
        fake_ab = ops.stop_gradient(fake_ab)
        d_in_real = self.concat((l, ab_gt))
        pred_real = self.net_d(d_in_real)
        d_in_fake = self.concat((l, fake_ab))
        pred_fake = self.net_d(d_in_fake)
        dummy_rgb = self.zeros_like(real_rgb)
        _, loss_d = self.loss_fn(pred_real, pred_fake, ab_gt, ab_gt, mask, real_rgb, dummy_rgb)
        return loss_d

    def construct(self, l, hint, mask, ab_gt):
        real_rgb = self.concat((l, ab_gt))
        loss_d = self.d_loss_func(l, hint, mask, ab_gt, real_rgb)
        grads_d = self.grad(self.d_loss_func, self.optimizer_d.parameters)(l, hint, mask, ab_gt, real_rgb)
        d_update = self.optimizer_d(grads_d)
        l_depend = self.depend(l, d_update)
        loss_g = self.g_loss_func(l_depend, hint, mask, ab_gt, real_rgb)
        grads_g = self.grad(self.g_loss_func, self.optimizer_g.parameters)(l_depend, hint, mask, ab_gt, real_rgb)
        g_update = self.optimizer_g(grads_g)
        loss_g = self.depend(loss_g, g_update)
        loss_d = self.depend(loss_d, d_update)
        g_input = self.concat((l, hint, mask))
        fake_ab = self.net_g(g_input)
        fake_ab = self.depend(fake_ab, g_update)
        return loss_g, loss_d, fake_ab

def get_lr(base_lr, epochs, steps_per_epoch, start_epoch=0):
    lrs = []
    half_epochs = epochs // 2
    # 补齐之前的 epoch (虽然不训练，但要占位以保持 LR 曲线一致)
    for _ in range(start_epoch * steps_per_epoch):
        lrs.append(0.0) 
        
    for epoch in range(start_epoch, epochs):
        lr = base_lr
        if epoch >= half_epochs:
            lr = base_lr * (1.0 - (epoch - half_epochs) / (epochs - half_epochs))
        for _ in range(steps_per_epoch):
            lrs.append(lr)
    return ms.Tensor(lrs[start_epoch * steps_per_epoch:], ms.float32)

def train(args):
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

    train_path = os.path.join(args.data_root, args.dataset_name, 'train_set')
    if args.dataset_name.lower() == 'ncd':
        train_path = os.path.join(args.data_root, 'NCD', 'original')

    dataset = create_dataset(train_path, dataset_name=args.dataset_name, split='train', batch_size=args.batch_size)
    steps_per_epoch = dataset.get_dataset_size()

    net_g = UNetGenerator(input_nc=4, output_nc=2)
    net_d = PatchGANDiscriminator(input_nc=3)

    # === 断点续训逻辑 ===
    if args.resume_ckpt:
        print(f"🔄 Resuming training from: {args.resume_ckpt}")
        try:
            param_dict = load_checkpoint(args.resume_ckpt)
            load_param_into_net(net_g, param_dict)
            print("✅ Generator weights loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load resume checkpoint: {e}")
            return
    else:
        print("🆕 Starting training from scratch.")

    # 学习率策略调整
    lr_g = get_lr(2e-4, args.epochs, steps_per_epoch, start_epoch=args.start_epoch)
    lr_d = get_lr(2e-4, args.epochs, steps_per_epoch, start_epoch=args.start_epoch)

    opt_g = nn.Adam(net_g.trainable_params(), learning_rate=lr_g, beta1=0.5, beta2=0.999)
    opt_d = nn.Adam(net_d.trainable_params(), learning_rate=lr_d, beta1=0.5, beta2=0.999)

    loss_fn = UserGuidedLoss(vgg_ckpt_path=args.vgg_path)
    train_step = GAN_TrainOneStepCell(net_g, net_d, opt_g, opt_d, loss_fn)

    net_g.set_train()
    net_d.set_train()

    print(f"🚀 Start training from Epoch {args.start_epoch + 1} to {args.epochs}...")

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        loss_g_epoch = []
        loss_d_epoch = []

        for i, data in enumerate(dataset.create_dict_iterator()):
            l = data['l_data']
            ab_gt = data['ab_gt']
            hint = data['ab_hint']
            mask = data['mask']

            loss_g, loss_d, fake_ab = train_step(l, hint, mask, ab_gt)

            loss_g_epoch.append(loss_g.asnumpy())
            loss_d_epoch.append(loss_d.asnumpy())

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{args.epochs}] Step [{i}/{steps_per_epoch}] "
                      f"G_Loss: {loss_g.asnumpy():.4f} D_Loss: {loss_d.asnumpy():.4f}")

        mean_g = np.mean(loss_g_epoch)
        mean_d = np.mean(loss_d_epoch)
        print(f"Epoch {epoch + 1} Cost: {time.time() - start_time:.2f}s | Mean G: {mean_g:.4f} | Mean D: {mean_d:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            save_checkpoint(net_g, os.path.join(args.train_url, f"net_g_{epoch + 1}.ckpt"))
            visualize_result(l, fake_ab, ab_gt, os.path.join(args.train_url, f"vis_{epoch + 1}.png"))

def visualize_result(l, fake_ab, real_ab, save_path):
    l_np = l[0].asnumpy()
    fake_ab_np = fake_ab[0].asnumpy()
    real_ab_np = real_ab[0].asnumpy()
    img_fake = lab_to_rgb(l_np, fake_ab_np)
    img_real = lab_to_rgb(l_np, real_ab_np)
    concat = np.hstack((img_real, img_fake))
    concat = (concat * 255).astype(np.uint8)
    concat = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, concat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, default='coco')
    parser.add_argument('--train_url', type=str, default='./output')
    parser.add_argument('--vgg_path', type=str, default='./checkpoints/vgg16.ckpt')
    parser.add_argument('--device_target', type=str, default='CPU')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    # 新增参数
    parser.add_argument('--resume_ckpt', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch number to start from')
    
    args = parser.parse_args()

    if not os.path.exists(args.train_url):
        os.makedirs(args.train_url)
    train(args)
