import torch
import torch.nn as nn

import math

from functools import reduce

__all__ = [
    'SENet',
    'SKNet',
    'SCSE',
    'ECANet',
    'CBAM_Block',
    'Attention'
]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            SENet 注意力机制
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class SENet(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(SENet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.in_channels, int(self.in_channels / ratio), bias=False)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(int(self.in_channels / ratio), self.in_channels, bias=False)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        output = self.fgp(x)
        output = output.view(b, c)
        output = self.fc1(output)
        output = self.act1(output)
        output = self.fc2(output)
        output = self.act2(output)
        output = output.view(b, c, 1, 1)
        return torch.multiply(x, output)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            CKNet 注意力机制
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class SKNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        """
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        """
        super(SKNet, self).__init__()
        d = max(in_channels // r, L)  
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList() 
        for i in range(M):
           
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1) 
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)  
        self.softmax = nn.Softmax(dim=1) 
    def forward(self, input):
        batch_size = input.size(0)
        output = []
        for i, conv in enumerate(self.conv):
            output.append(conv(input))
        U = reduce(lambda x, y: x + y, output)  
        s = self.global_pool(U)  
        z = self.fc1(s)
        a_b = self.fc2(z) 
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1) 
        a_b = self.softmax(a_b) 
        a_b = list(a_b.chunk(self.M, dim=1))  
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))  
        V = list(map(lambda x, y: x * y, output,
                     a_b))  
        V = reduce(lambda x, y: x + y,
                   V)  
        return V 


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            SCSE 注意力机制
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class SCSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCSE, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            ECANet 注意力机制
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            CBAM 双域注意力机制
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class ChannelAttention(nn.Module):
    """
    CBAM混合注意力机制的通道注意力
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            # nn.Linear(in_planes, in_planes // ratio, bias=False),
            # nn.ReLU(),
            # nn.Linear(in_planes // ratio, in_planes, bias=False)

            # 利用1x1卷积代替全连接
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class CBAM_Block(nn.Module):
    """
    CBAM混合注意力机制
    """

    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM_Block, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            注意力机制
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSE(**params)
        elif name == "senet":
            self.attention = SENet(**params)
        elif name == "sknet":
            self.attention = SKNet(**params)
        elif name == "ecanet":
            self.attention = ECANet(**params)
        elif name == "cbam":
            self.attention = CBAM_Block(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


