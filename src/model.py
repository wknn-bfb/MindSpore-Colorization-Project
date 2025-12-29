import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal, XavierUniform

def init_weights(net, init_type='normal', gain=0.02):
    """
    权重初始化函数
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(
                    mindspore.common.initializer.initializer(Normal(gain), cell.weight.shape, cell.weight.dtype))
            elif init_type == 'xavier':
                cell.weight.set_data(
                    mindspore.common.initializer.initializer(XavierUniform(gain=gain), cell.weight.shape,
                                                             cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(mindspore.common.initializer.initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            cell.gamma.set_data(mindspore.common.initializer.initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(mindspore.common.initializer.initializer('zeros', cell.beta.shape, cell.beta.dtype))


class ConvBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, is_transpose=False,
                 use_dropout=False):
        super(ConvBlock, self).__init__()
        layers = []
        if is_transpose:
            layers.append(
                nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding,
                                   has_bias=True))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='pad', padding=padding,
                                    has_bias=True))

        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(alpha=0.2) if not is_transpose else nn.ReLU())
        if use_dropout:
            # 【修复】MindSpore 1.7 使用 keep_prob
            layers.append(nn.Dropout(keep_prob=0.5))
        self.block = nn.SequentialCell(layers)

    def construct(self, x):
        return self.block(x)


class UNetGenerator(nn.Cell):
    def __init__(self, input_nc=4, output_nc=2, ngf=64):
        super(UNetGenerator, self).__init__()
        # Encoder
        self.down1 = nn.Conv2d(input_nc, ngf, 4, 2, pad_mode='pad', padding=1, has_bias=True)
        self.down2 = ConvBlock(ngf, ngf * 2)
        self.down3 = ConvBlock(ngf * 2, ngf * 4)
        self.down4 = ConvBlock(ngf * 4, ngf * 8)
        self.down5 = ConvBlock(ngf * 8, ngf * 8)
        self.down6 = ConvBlock(ngf * 8, ngf * 8)
        self.down7 = ConvBlock(ngf * 8, ngf * 8)
        self.down8 = ConvBlock(ngf * 8, ngf * 8, use_dropout=False)

        # Decoder
        self.up1 = ConvBlock(ngf * 8, ngf * 8, is_transpose=True, use_dropout=True)
        self.up2 = ConvBlock(ngf * 8 * 2, ngf * 8, is_transpose=True, use_dropout=True)
        self.up3 = ConvBlock(ngf * 8 * 2, ngf * 8, is_transpose=True, use_dropout=True)
        self.up4 = ConvBlock(ngf * 8 * 2, ngf * 8, is_transpose=True)
        self.up5 = ConvBlock(ngf * 8 * 2, ngf * 4, is_transpose=True)
        self.up6 = ConvBlock(ngf * 4 * 2, ngf * 2, is_transpose=True)
        self.up7 = ConvBlock(ngf * 2 * 2, ngf, is_transpose=True)

        self.final = nn.SequentialCell([
            nn.Conv2dTranspose(ngf * 2, output_nc, 4, 2, pad_mode='pad', padding=1, has_bias=True),
            nn.Tanh()
        ])
        self.lrelu = nn.LeakyReLU(alpha=0.2)
        
        # 【修复】MindSpore 1.7 必须先定义算子类
        self.concat = ops.Concat(axis=1)

        # 初始化权重
        init_weights(self, 'normal', 0.02)

    def construct(self, x):
        d1 = self.lrelu(self.down1(x))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        # 【修复】使用 self.concat 替代 ops.concat
        u1 = self.concat((u1, d7))
        u2 = self.up2(u1)
        u2 = self.concat((u2, d6))
        u3 = self.up3(u2)
        u3 = self.concat((u3, d5))
        u4 = self.up4(u3)
        u4 = self.concat((u4, d4))
        u5 = self.up5(u4)
        u5 = self.concat((u5, d3))
        u6 = self.up6(u5)
        u6 = self.concat((u6, d2))
        u7 = self.up7(u6)
        u7 = self.concat((u7, d1))

        return self.final(u7)


class PatchGANDiscriminator(nn.Cell):
    def __init__(self, input_nc=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.SequentialCell([
            nn.Conv2d(input_nc, ndf, 4, 2, pad_mode='pad', padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            ConvBlock(ndf, ndf * 2, stride=2),
            ConvBlock(ndf * 2, ndf * 4, stride=2),
            ConvBlock(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, 4, 1, pad_mode='pad', padding=1, has_bias=True)
        ])
        init_weights(self, 'normal', 0.02)

    def construct(self, x):
        return self.model(x)
